import os
import numpy as np
import math
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt


# Define 2D particle structure
@dataclass
class Point2D:
    def __init__(self, x=0.0, y=0.0, u=0.0, v=0.0, rho=0.0, mu=0.0, 
                 properties=0, pressure=0.0, mass=0.0, 
                 acceleration_x=0.0, acceleration_y=0.0, id=0):
        self.x = x  # x position
        self.y = y  # y position
        self.u = u  # x velocity
        self.v = v  # y velocity
        self.rho = rho  # density
        self.mu = mu  # viscosity
        self.properties = properties  # 0: fluid particle, 1: mirror boundary particle, 2: periodic boundary particle,3
        self.pressure = pressure
        self.mass = mass
        self.acceleration_x = acceleration_x
        self.acceleration_y = acceleration_y
        self.id = id  # Particle ID
        self.divergence_r = 0

class SPH:
    # Static class variables
    sharedStaticData = -1
    countFluid = 0
    countVirtual = 0
    MinGridX = 0
    MaxGridX = 0
    MinGridy = 0
    MaxGridy = 0
    
    def __init__(self):
        self.Dimension = 2
        self.print_time = 1
        self.PI = math.pi
        
        # Domain and particle parameters
        self.initialX = 0.0
        self.endX = 0.39
        self.initialY = 0.0
        self.endY = 0.39
        self.spacing = 0.01
        self.l = self.endX - self.initialX
        
        # SPH parameters
        self.h = 1.3 * self.spacing  # Smoothing length
        self.h_coefficient = 3.0  # Grid coefficient
        self.fx = 0.0  # x-direction acceleration
        self.gravity = 9.8
        self.C0 = 18.187  # Reference sound speed
        self.particle_number = int(((self.endY - self.initialY) / self.spacing) * ((self.endX - self.initialX) / self.spacing))
        self.density_0 = 1000.0  # Initial density
        self.mu_0 = 0.1  # Viscosity coefficient
        
        self.gamma = 7.0
        self.t_damp = 0.35  # Damping time
        self.B = pow(self.C0, 2.0) * self.density_0 / self.gamma
        self.alpha = 1.0
        self.beta = 0.0
        
        # Initialize data structures
        self.particle = Point2D(0.0, 0.0)  # Create a Point2D object to store particle position
        self.particlePositions = []  # Vector to store particle positions
        self.grid = defaultdict(list)  # Multimap equivalent in Python (grid cells to particle IDs)
        self.interactionPairs = defaultdict(list)  # Store interacting particle pairs
        self.particleMap = {}  # Map particle IDs to indices
        self.matricesMap = {}  # Map to store matrices for particles
        
        # Initialize vectors for particle properties
        self.VX = []  # x-direction acceleration
        self.VY = []  # y-direction acceleration
        self.Morris_x = []  # Morris viscosity x-component
        self.Morris_y = []  # Morris viscosity y-component
        
        # VLX algorithm parameters
        self.vlx_X = 7  # Tentative number of time steps to keep the Verlet list 
        self.vlx_max_vel = 0.0  # Maximum velocity in the system
        self.vlx_init_positions = {}  # Store initial positions when building the Verlet list
        self.vlx_delta_h = 0.0  # Distance threshold for rebuilding the Verlet list
        self.vlx_last_rebuild = 0  # Time step of the last rebuild

    def creatparticle(self):
        """Create fluid particles"""
        particle_id = 0
        # Generate fluid particles in the domain
        for x in np.arange(self.initialX + self.spacing/2.0, self.endX + self.spacing/2.0, self.spacing):
            for y in np.arange(self.initialY + self.spacing/2.0, self.endY + self.spacing/2.0, self.spacing):
                # Create a new particle
                self.particle = Point2D()
                self.particle.x = x
                self.particle.y = y
                self.particle.u = 0.0
                self.particle.v = 0.0
                self.particle.rho = self.density_0
                self.particle.properties = 0
                self.particle.mu = self.mu_0
                depth = self.endY - y
                self.particle.pressure = self.particle.rho * self.gravity * depth
                self.particle.mass = self.particle.rho * self.spacing * self.spacing
                self.particle.acceleration_x = 0.0
                self.particle.acceleration_y = -self.gravity
                self.particle.id = particle_id
                # Add particle to the list
                self.particlePositions.append(self.particle)
                particle_id += 1
                SPH.countFluid += 1
                #print(f"pariticle number :{particle_id} position:  x:{x},y:{y}")
                
        print(f"fluid number:{SPH.countFluid}")
            
    def grids(self):
        """Assign particles to grid cells"""
        min_grid_x = float('inf')
        max_grid_x = float('-inf')
        min_grid_y = float('inf')
        max_grid_y = float('-inf')
        
        # Clear the grid before assigning particles
        self.grid.clear()
        
        # Assign fluid particles to grid cells
        for i in range(SPH.countFluid):
            grid_x = math.floor(self.particlePositions[i].x / (self.h_coefficient * self.h))
            grid_y = math.floor(self.particlePositions[i].y / (self.h_coefficient * self.h))
            grid_key = (grid_x, grid_y)
            self.grid[grid_key].append(i)
            
            # Update min and max grid indices
            min_grid_x = min(min_grid_x, grid_x)
            max_grid_x = max(max_grid_x, grid_x)
            min_grid_y = min(min_grid_y, grid_y)
            max_grid_y = max(max_grid_y, grid_y)
            
        # Update class variables
        SPH.MinGridX = min_grid_x
        SPH.MaxGridX = max_grid_x
        SPH.MinGridy = min_grid_y
        SPH.MaxGridy = max_grid_y
        
    def Virtual_particle(self):
        """Create virtual particles for all boundaries including top free surface"""
        particle_id = len(self.particlePositions)
        
        # Generate top and bottom boundary virtual particles
        for j in range(SPH.MinGridy, SPH.MaxGridy + 1):
            for i in range(SPH.MinGridX, SPH.MaxGridX + 1):
                grid_key = (i, j)
                
                # For each grid cell
                if SPH.MinGridX - 1 <= i <= SPH.MaxGridX + 1:
                    # Bottom boundary
                    if j == SPH.MinGridy:
                        for particle_idx in self.grid.get(grid_key, []):
                            
                            if particle_idx >= SPH.countFluid:
                                break
                                
                            # Create mirror particle
                            self.particle = Point2D()                                                     
                            self.particle.x = self.particlePositions[particle_idx].x
                            self.particle.y = 2.0 * self.initialY - self.particlePositions[particle_idx].y
                            self.particle.u = -self.particlePositions[particle_idx].u
                            self.particle.v = -self.particlePositions[particle_idx].v
                            self.particle.mass = self.particlePositions[particle_idx].mass
                            self.particle.mu = self.particlePositions[particle_idx].mu
                            self.particle.properties = 1  # Mirror boundary particle
                            self.particle.rho = self.particlePositions[particle_idx].rho
                            depth = self.endY - self.particle.y
                            self.particle.pressure = self.particle.rho * self.gravity * depth
                            self.particle.acceleration_x=0.0
                            self.particle.acceleration_y = -self.gravity
                            self.particle.id=particle_id

                            self.particlePositions.append(self.particle)
                            self.particleMap[particle_id] = particle_idx
                            self.grid[(i, j-1)].append(particle_id)
                            particle_id += 1
        
        # Generate left and right boundary virtual particles with mirror boundaries
        for j in range(SPH.MinGridy-1, SPH.MaxGridy+2):
            for i in range(SPH.MinGridX, SPH.MaxGridX+1):
                grid_key = (i, j)
                
                # For each grid cell within the domain plus virtual layer
                if SPH.MinGridy-1 <= j <= SPH.MaxGridy+1:
                    # Right boundary
                    if i == SPH.MaxGridX:
                        for particle_idx in self.grid.get(grid_key, []):
                            
                            
                            self.particle = Point2D(
                                x=2.0 * self.endX - self.particlePositions[particle_idx].x,
                                y=self.particlePositions[particle_idx].y,
                                u=-self.particlePositions[particle_idx].u,
                                v=self.particlePositions[particle_idx].v,
                                mass=self.particlePositions[particle_idx].mass,
                                mu=self.particlePositions[particle_idx].mu,
                                properties=1,  
                                rho=self.particlePositions[particle_idx].rho,
                                pressure=self.particlePositions[particle_idx].pressure,
                                acceleration_x=0.0,
                                acceleration_y=-self.gravity,
                                id=particle_id
                            )
                            self.particlePositions.append(self.particle)
                            self.particleMap[particle_id] = particle_idx
                            self.grid[(i+1, j)].append(particle_id)
                            particle_id += 1
                            
                    # Left boundary
                    if i == SPH.MinGridX:
                        for particle_idx in self.grid.get(grid_key, []):
                            
                            # Create mirror boundary particle instead of periodic
                            self.particle = Point2D(
                                x=2.0 * self.initialX - self.particlePositions[particle_idx].x,
                                y=self.particlePositions[particle_idx].y,
                                u=-self.particlePositions[particle_idx].u,
                                v=self.particlePositions[particle_idx].v,
                                mass=self.particlePositions[particle_idx].mass,
                                mu=self.particlePositions[particle_idx].mu,
                                properties=1,  # Changed to mirror boundary particle
                                rho=self.particlePositions[particle_idx].rho,
                                pressure=self.particlePositions[particle_idx].pressure,
                                acceleration_x=0.0,
                                acceleration_y=-self.gravity,
                                id=particle_id
                            )
                            self.particlePositions.append(self.particle)
                            self.particleMap[particle_id] = particle_idx
                            self.grid[(i-1, j)].append(particle_id)
                            particle_id += 1
    
    def InteractingParticle(self):
        """
        VLX Algorithm: Improved Verlet list neighbor search
        
        This method implements the VLX algorithm for neighbor searching,
        which dynamically determines when to rebuild the neighbor list
        based on particle movement.
        """
        
        # Only rebuild the Verlet list if necessary
        should_rebuild = False
        
        # Initial build of Verlet list
        if not self.interactionPairs or SPH.sharedStaticData == 0:
            should_rebuild = True
        else:
            # Check if any particle has moved more than Δh/2
            for i in range(SPH.countFluid):
                if i in self.vlx_init_positions:
                    dx = self.particlePositions[i].x - self.vlx_init_positions[i][0]
                    dy = self.particlePositions[i].y - self.vlx_init_positions[i][1]
                    distance_moved = math.sqrt(dx*dx + dy*dy)
                    
                    if distance_moved > self.vlx_delta_h/2:
                        should_rebuild = True
                        break
                else:

                    should_rebuild = True
                    break
        
        # If  need to rebuild the Verlet list
        if should_rebuild:
            # Clear the existing list
            self.interactionPairs.clear()
            self.vlx_init_positions.clear()
            
            # Update max velocity in the system
            self.vlx_max_vel = 0.0
            for i in range(SPH.countFluid):
                vel_mag = math.sqrt(self.particlePositions[i].u**2 + self.particlePositions[i].v**2)
                self.vlx_max_vel = max(self.vlx_max_vel, vel_mag)
            
            # Calculate Δh = 2 * Vmax * X * dt
            # This is the safety distance for the Verlet list
            dt = 0.00005  #  time step 
            self.vlx_delta_h = 2.0 * self.vlx_max_vel * self.vlx_X * dt
            
            # Set the search radius to 2H = 2h + Δh
            search_radius = 2.0 * self.h + self.vlx_delta_h
            
            # Build the neighbor list using the grid
            for j in range(SPH.MinGridy, SPH.MaxGridy + 1):
                for i in range(SPH.MinGridX, SPH.MaxGridX + 1):
                    grid_key = (i, j)
                    
                    # For each particle in the current grid cell
                    for particle_idx in self.grid.get(grid_key, []):
                        if 0 <= particle_idx < SPH.countFluid:
                            # Store initial position
                            self.vlx_init_positions[particle_idx] = (
                                self.particlePositions[particle_idx].x,
                                self.particlePositions[particle_idx].y
                            )
                            
                            # Search for neighbors in a 3x3 grid neighborhood
                            for m in range(i - 1, i + 2):
                                for n in range(j - 1, j + 2):
                                    neighbor_grid = (m, n)
                                    for neighbor_idx in self.grid.get(neighbor_grid, []):
                                        # Skip self-interaction
                                        if neighbor_idx != particle_idx:
                                            r = self.calculateDistance(
                                                self.particlePositions[particle_idx], 
                                                self.particlePositions[neighbor_idx]
                                            )
                                            # Use the enhanced search radius (2h + Δh)
                                            if r <= search_radius:
                                                self.interactionPairs[particle_idx].append(neighbor_idx)
            
            # Update the last rebuild time step
            self.vlx_last_rebuild = SPH.sharedStaticData
            print(f"VLX: Rebuilding neighbor list at step {SPH.sharedStaticData}, max vel: {self.vlx_max_vel:.5f}, Δh: {self.vlx_delta_h:.5f}")
   
    def Density(self, delta_time):
        
        # Sort keys for consistent processing order
        keys = sorted(list(self.interactionPairs.keys()))
        
        for i in keys:
            neighbors = self.interactionPairs[i]  # Get neighbors of particle i
            density_dt = 0.0
            
            for j in neighbors:

                if i==j:
                    continue

                r = self.calculateDistance(self.particlePositions[i], self.particlePositions[j])  # Calculate distance
                R = r / self.h  # Calculate relative distance
                
                dx = self.particlePositions[i].x - self.particlePositions[j].x
                dy = self.particlePositions[i].y - self.particlePositions[j].y
                
                #epsilon = 1e-17  # Small value to prevent division by zero
                dwx = self.kernel_cubic_core(self.h, R) * dx / r
                dwy = self.kernel_cubic_core(self.h, R) * dy / r
                
                Vxij = self.particlePositions[i].u - self.particlePositions[j].u
                Vyij = self.particlePositions[i].v - self.particlePositions[j].v
                
                density_dt += self.particlePositions[j].mass * (Vxij * dwx + Vyij * dwy)                

            # Update density
            self.particlePositions[i].rho += density_dt * delta_time
               
        # Increment static counter
        SPH.sharedStaticData += 1
        print(SPH.sharedStaticData)

    def Pressure(self):
        """Calculate particle pressure"""
        for i in range(self.countFluid):
            # Calculate pressure using equation of state
            temp1 = (self.particlePositions[i].rho / self.density_0) ** self.gamma
            self.particlePositions[i].pressure = self.B * (temp1 - 1.0)

    def mirror(self):
        """Update mirror particles' pressure and density based on their boundary position"""
        for j in range(self.countFluid, len(self.particlePositions)):
            if j in self.particleMap:
                i = self.particleMap[j]  # Get corresponding real particle index
                mirror_particle = self.particlePositions[j]
                real_particle = self.particlePositions[i]
                
                # Determine boundary type based on position
                # Bottom boundary mirror particles
                if mirror_particle.y < self.initialY:
                    mirror_particle.rho = real_particle.rho
                    mirror_particle.pressure = real_particle.pressure
                
                # Right boundary mirror particles
                elif mirror_particle.x > self.endX:
                    mirror_particle.rho = real_particle.rho
                    mirror_particle.pressure = real_particle.pressure
                
                # Left boundary mirror particles
                elif mirror_particle.x < self.initialX:
                    mirror_particle.rho = real_particle.rho
                    mirror_particle.pressure = real_particle.pressure
                
                # For other mirror particles (if any)

                else:
                    print("出现不存在的边界粒子")
                    break

    def resetAcceleration(self):
        """resetAcceleration"""
        for i in range(SPH.countFluid):
            self.particlePositions[i].acceleration_x = 0.0
            self.particlePositions[i].acceleration_y = -self.gravity  #

    def visc(self):
        """Calculate artificial viscosity"""        
        # Sort keys for consistent processing order
        keys = sorted(list(self.interactionPairs.keys()))      
        for i in keys:
            neighbors = self.interactionPairs[i]  # Get neighbors of particle i
            visc_x, visc_y = 0.0, 0.0            
            for j in neighbors:  
                if i==j:
                    continue                  
                # Calculate viscosity contribution
                dx = self.particlePositions[i].x - self.particlePositions[j].x
                dy = self.particlePositions[i].y - self.particlePositions[j].y
                r = self.calculateDistance(self.particlePositions[i], self.particlePositions[j])
                R = r / self.h                
                dwx = self.kernel_cubic_core(self.h, R) * dx / r 
                dwy = self.kernel_cubic_core(self.h, R) * dy / r 
                
                Vxij = self.particlePositions[i].u - self.particlePositions[j].u
                Vyij = self.particlePositions[i].v - self.particlePositions[j].v
                
                rr = dx**2 + dy**2
                average_density = (self.particlePositions[i].rho + self.particlePositions[j].rho) / 2.0
                vr = Vxij * dx + Vyij * dy
                
                varphi = (0.1 * self.h)**2
                lamada = self.h * vr / (rr + varphi)
                
                # Apply damping based on relative velocity
                if vr < 0:
                    damp = (-self.alpha * self.C0 * lamada) / average_density
                else:
                    damp = 0                    
                visc_x += damp * self.particlePositions[j].mass * dwx
                visc_y += damp * self.particlePositions[j].mass * dwy            
            # Store viscosity contributions
            self.particlePositions[i].acceleration_x += -visc_x
            self.particlePositions[i].acceleration_y += -visc_y
            """Save particle data to files"""
        
        if SPH.sharedStaticData % self.print_time == 0:
            # Create directory for data if it doesn't exist
            data_dir = "/home/jiaozi/HYDROSTATIC/Data/dubug_visc_1"
            os.makedirs(data_dir, exist_ok=True)
            
            # Create output file with timestep information
            filename = f"{data_dir}/AFTER visc particles_step_{SPH.sharedStaticData}.csv"
            
            # Write particle data to CSV file
            with open(filename, 'w') as fout:
                # Write header
                fout.write("ID,X_Position,Y_Position,acceleration_x,acceleration_y\n")
                
                # Write particle data
                for particle in self.particlePositions:
                    if particle.id < SPH.countFluid:
                        fout.write(f"{particle.id},{particle.x},{particle.y},{particle.acceleration_x},{particle.acceleration_y}\n")
        

    def PressureGradient(self):
        """Calculate pressure gradient force"""
        # Sort keys for consistent processing order
        keys = sorted(list(self.interactionPairs.keys()))
        
        for i in keys:
            neighbors = self.interactionPairs[i]  # Get neighbors of particle i
            dv_dx, dv_dy = 0.0, 0.0  # Velocity derivatives
            
            for j in neighbors:
                if i==j:
                    continue
                dx = self.particlePositions[i].x - self.particlePositions[j].x
                dy = self.particlePositions[i].y - self.particlePositions[j].y
                r = self.calculateDistance(self.particlePositions[i], self.particlePositions[j])
                R = r / self.h
                
                dwx = self.kernel_cubic_core(self.h, R) * dx / r 
                dwy = self.kernel_cubic_core(self.h, R) * dy / r 
                
                # Calculate pressure terms
                Pi = self.particlePositions[i].pressure / (self.particlePositions[i].rho**2)
                Pj = self.particlePositions[j].pressure / (self.particlePositions[j].rho**2)
                
                px = -(Pi + Pj) * dwx
                py = -(Pi + Pj) * dwy
                
                dv_dx += self.particlePositions[j].mass * px
                dv_dy += self.particlePositions[j].mass * py
            
            # Update accelerations
            self.particlePositions[i].acceleration_x += dv_dx
            self.particlePositions[i].acceleration_y += dv_dy
            #print(f'After PressureGradient ,acceleration_y{i}={self.particlePositions[i].acceleration_y}')

        if SPH.sharedStaticData % self.print_time == 0:
            # Create directory for data if it doesn't exist
            data_dir = "/home/jiaozi/HYDROSTATIC/Data/dubug_PressureGradient_1"
            os.makedirs(data_dir, exist_ok=True)
            
            # Create output file with timestep information
            filename = f"{data_dir}/AFTER PressureGradient particles_step_{SPH.sharedStaticData}.csv"
            
            # Write particle data to CSV file
            with open(filename, 'w') as fout:
                # Write header
                fout.write("ID,X_Position,Y_Position,acceleration_x,acceleration_y\n")
                
                # Write particle data
                for particle in self.particlePositions:
                    if particle.id < SPH.countFluid:
                        fout.write(f"{particle.id},{particle.x},{particle.y},{particle.acceleration_x},{particle.acceleration_y}\n")

    def Morris(self):
        """Calculate Morris viscosity"""
        # Sort keys for consistent processing order
        keys = sorted(list(self.interactionPairs.keys()))       
        for i in keys:
            neighbors = self.interactionPairs[i]  # Get neighbors of particle i
            VX, VY = 0.0, 0.0            
            for j in neighbors:

                if i==j:
                    continue

                dx = self.particlePositions[i].x - self.particlePositions[j].x
                dy = self.particlePositions[i].y - self.particlePositions[j].y
                r = self.calculateDistance(self.particlePositions[i], self.particlePositions[j])
                R = r / self.h
                
                dwx = self.kernel_cubic_core(self.h, R) * dx / r
                dwy = self.kernel_cubic_core(self.h, R) * dy / r 
                
                Vxij = self.particlePositions[i].u - self.particlePositions[j].u
                Vyij = self.particlePositions[i].v - self.particlePositions[j].v
                
                rr = r**2
                # Calculate viscosity contribution
                rw = dx * dwx + dy * dwy
                
                mui = self.particlePositions[i].mu * self.particlePositions[i].rho
                muj = self.particlePositions[j].mu * self.particlePositions[j].rho
                mu = mui + muj
                rho = self.particlePositions[i].rho * self.particlePositions[j].rho                
                xij = rr
                tmep = self.particlePositions[j].mass * mu * rw / (rho * xij)                
                # Accumulate viscosity effect
                VX += tmep * Vxij
                VY += tmep * Vyij            
            # Update accelerations with Morris viscosity
            self.particlePositions[i].acceleration_x += VX
            self.particlePositions[i].acceleration_y += VY

    def Damping(self, total_time):
        """Calculate damping factor for particle velocity"""
        zeta = 1
        if total_time < self.t_damp:
            a = total_time / self.t_damp
            b = np.sin(self.PI * (-0.5 + a))
            zeta = 0.5 * (b + 1)
        return zeta
            


    def update(self, deltaTime,total_time):
        """Update particle positions and velocities"""
        
        #zeta = self.Damping(total_time)
        
        for i in range(SPH.countFluid):
            # Update velocities with acceleration
            self.particlePositions[i].u += self.particlePositions[i].acceleration_x * deltaTime
            self.particlePositions[i].v += self.particlePositions[i].acceleration_y * deltaTime
            
            # Apply damping to velocities
            #self.particlePositions[i].u *= zeta
            #self.particlePositions[i].v *= zeta
        

        for i in range(SPH.countFluid):
            self.particlePositions[i].x += self.particlePositions[i].u * deltaTime
            self.particlePositions[i].y += self.particlePositions[i].v * deltaTime
        
    def calculateDistance(self, particle_a, particle_b):
        """Calculate Euclidean distance between two particles"""
        dx = particle_a.x - particle_b.x
        dy = particle_a.y - particle_b.y
        return math.sqrt(dx*dx + dy*dy)

    def kernel_cubic(self, h, R):
        """Cubic spline kernel function"""
        alpha = 10.0/ (7.0 * self.PI * h**2)
        if 0 <= R < 1:
            return alpha * (2.0/3.0 -  R**2 + 0.5 * R**3)
        elif 1 <= R < 2:
            return alpha * 1.0/6.0 * (2 - R)**3
        else:
            return 0.0

    def kernel_cubic_core(self, h, R):
        """Derivative of cubic spline kernel"""
        alpha = 10.0/(7.0 * self.PI * h**2)
        if 0 <= R < 1:
            return alpha * (-3 * R + 2.25 * R**2) / h
        elif 1 <= R < 2:
            return -alpha * 0.75 * (2 - R)**2 / h
        else:
            return 0.0
        
    def kernel_gaussian(self, h, R):
        """Gaussian kernel function"""
        alpha = 1.0 / (self.PI * h**2)
        return alpha * math.exp(-R**2)

    def kernel_gaussian_core(self, h, R):
        """Derivative of Gaussian kernel"""
        alpha = 1.0 / (self.PI * h**2)
        return -2 * alpha * R * math.exp(-R**2) / h

    def kernel_wendland_c2(self, h, R):
        """Wendland C2 kernel function"""
        alpha = 7.0 / (4.0 * self.PI * h**2)
        if 0 <= R < 1:
            return alpha * (1 - R)**4 * (1 + 4 * R)
        else:
            return 0.0

    def kernel_wendland_c2_core(self, h, R):
        """Derivative of Wendland C2 kernel"""
        alpha = 7.0 / (4.0 * self.PI * h**2)
        if 0 <= R < 1:
            return -20 * alpha * R * (1 - R)**3 / h
        else:
            return 0.0
        
    def kernel_wendland_c4(self, h, R):
        """Wendland C4 kernel function"""
        alpha = 9.0 / (self.PI * h**2)
        if 0 <= R < 1:
            return alpha * (1 - R)**6 * (35 * R**2 + 18 * R + 3)
        else:
            return 0.0

    def kernel_wendland_c4_core(self, h, R):
        """Derivative of Wendland C4 kernel"""
        alpha = 9.0 / (self.PI * h**2)
        if 0 <= R < 1:
            return -alpha * (1 - R)**5 * (245 * R**2 + 126 * R + 15) / h
        else:
            return 0.0

    def kernel_wendland_c6(self, h, R):
        """Wendland C6 kernel function"""
        alpha = 495.0 / (32.0 * self.PI * h**2)
        if 0 <= R < 1:
            return alpha * (1 - R)**8 * (32 * R**3 + 25 * R**2 + 8 * R + 1)
        else:
            return 0.0

    def kernel_wendland_c6_core(self, h, R):
        """Derivative of Wendland C6 kernel"""
        alpha = 495.0 / (32.0 * self.PI * h**2)
        if 0 <= R < 1:
            return -alpha * (1 - R)**7 * (256 * R**3 + 231 * R**2 + 72 * R + 7) / h
        else:
            return 0.0
        
    def kernel_super_gaussian(self, h, R):
        """Super Gaussian kernel function"""
        alpha = 1.0 / (self.PI * h**2)
        return alpha * math.exp(-R**2) * (2 - R**2)

    def kernel_super_gaussian_core(self, h, R):
        """Derivative of Super Gaussian kernel"""
        alpha = 1.0 / (self.PI * h**2)
        return -2 * alpha * R * math.exp(-R**2) * (3 - R**2) / h

    def kernel_5core(self, h, R):
        """Quintic spline kernel derivative function core"""
        alpha = 0  # Normalization factor
        lamada = 0
        
        if self.Dimension == 2:
            alpha = 7.0 / (478.0 * self.PI * pow(h, 2.0))
        
        if 0 <= R < 1:
            lamada = alpha * (-5.0 * pow((3 - R), 4) + 30.0 * pow((2 - R), 4) - 75.0 * pow((1 - R), 4)) / h
        elif 1 <= R < 2:
            lamada = alpha * (-5.0 * pow((3 - R), 4) + 30.0 * pow((2 - R), 4)) / h
        elif 2 <= R < 3:
            lamada = alpha * (-5.0 * pow((3 - R), 4)) / h
        
        return lamada
    
    def kernel_5(self, h, R):
        """Quintic spline kernel function"""
        alpha = 0  # Normalization factor
        lamada = 0

        if self.Dimension == 2:
            alpha = 7.0 / (478.0 * self.PI * pow(h, 2.0))
        
        if 0 <= R < 1:
            lamada = alpha * (pow((3 - R), 5) - 6.0 * pow((2 - R), 5) + 15.0 * pow((1 - R), 5))
        elif 1 <= R < 2:
            lamada = alpha * (pow((3.0 - R), 5) - 6.0 * pow((2.0 - R), 5))
        elif 2 <= R < 3:
            lamada = alpha * pow((3.0 - R), 5)
        
        return lamada

    def plot_particles(self):
        """visualise fluid particle and virtual particle"""
        fluid_x, fluid_y = [], []
        virtual_x, virtual_y = [], []
        SAVE_DIR = "/home/jiaozi/HYDROSTATIC/visualization"  # Directory to save the plots
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        
        if SPH.sharedStaticData % self.print_time == 0:

            for i in range(len(self.particlePositions)):
                particle = self.particlePositions[i]
                if i < SPH.countFluid and particle.properties == 0:
                    fluid_x.append(particle.x)
                    fluid_y.append(particle.y)
                elif i >= SPH.countFluid and particle.properties == 1:
                    virtual_x.append(particle.x)
                    virtual_y.append(particle.y)

            plt.figure(figsize=(8, 6))
            plt.scatter(fluid_x, fluid_y, c='blue', s=10, label='Fluid Particles')
            plt.scatter(virtual_x, virtual_y, c='red', s=10, label='Virtual Particles')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Distribution of Fluid and Virtual Particles at step {SPH.sharedStaticData}")
            plt.savefig(os.path.join(SAVE_DIR, f'pressure_distribution_step_{SPH.sharedStaticData}.png'), dpi=300)
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.tight_layout()
            plt.show()   

    def divergence_distance(self):
        
        keys = list(self.interactionPairs.keys())
        keys.sort()
        epsilon = 1e-17

        for i in keys:
            neighbors = self.interactionPairs[i]
            divergence = 0.0
            xi = self.particlePositions[i].x
            yi = self.particlePositions[i].y

            for j in neighbors:
                xj = self.particlePositions[j].x
                yj = self.particlePositions[j].y

                dx = xj - xi
                dy = yj - yi
                r = math.sqrt(dx**2 + dy**2)
                R = r / self.h

               
                dW = self.kernel_cubic_core(self.h, R)
                gradW_x = - dW * dx / (r + epsilon)
                gradW_y = - dW * dy / (r + epsilon)

                # 点积：r_ji · ∇W_ij
                dot_product = dx * gradW_x + dy * gradW_y

                weight = self.particlePositions[j].mass / self.particlePositions[j].rho
                divergence += weight * dot_product

            self.particlePositions[i].divergence_r = divergence   
        
        # Create directory for data if it doesn't exist
        base_dir = "/home/jiaozi/HYDROSTATIC/Debug/divergence_distance/5.22.1"
        fluid_dir = f"{base_dir}/fluid"
        virtual_dir = f"{base_dir}/virtual"
        
        os.makedirs(fluid_dir, exist_ok=True)
        os.makedirs(virtual_dir, exist_ok=True)
        
        # Create directory for data if it doesn't exist
        fluid_filename = f"{fluid_dir}/particles_step_{SPH.sharedStaticData}.csv"
        # Create output file with timestep information
        virtual_filename = f"{virtual_dir}/particles_step_{SPH.sharedStaticData}.csv"
        
        if SPH.sharedStaticData % self.print_time == 0:
            # Write particle data to CSV file
            with open(fluid_filename, 'w') as fout:
                # Write header
                fout.write("ID,x_position,y_position,divergence_distance\n")
                
                # Write particle data
                for particle in self.particlePositions:
                    if particle.id < SPH.countFluid:  # fluid
                        fout.write(f"{particle.id},{particle.x},{particle.y},{particle.divergence_r}\n")
            
            # Write particle data
            with open(virtual_filename, 'w') as fout:
                # Write particle data to CSV file
                fout.write("ID,x_position,y_position,divergence_distance,original_particle_id\n")
                
                # Write particle data
                for particle in self.particlePositions:
                    if particle.id >= SPH.countFluid:  # virtual
                        original_id = self.particleMap.get(particle.id, -1)
                        fout.write(f"{particle.id},{particle.x},{particle.y},{particle.divergence_r},{original_id}\n")
            
            print(f"计算完成: 流体粒子数据保存至 {fluid_filename}, 虚拟粒子数据保存至 {virtual_filename}")

    def erase(self):
        """Clear virtual particles and grid data"""
        self.grid.clear()
        self.particleMap.clear()
        
        # Remove virtual particles
        if len(self.particlePositions) > SPH.countFluid:
            self.particlePositions = self.particlePositions[:SPH.countFluid]

    def print(self):
        """Save particle data to files"""
        if SPH.sharedStaticData % self.print_time == 0:
            # Create directory for data if it doesn't exist
            data_dir = "/home/jiaozi/HYDROSTATIC/Data/5.22.2"
            os.makedirs(data_dir, exist_ok=True)
            
            # Create output file with timestep information
            filename = f"{data_dir}/particles_step_{SPH.sharedStaticData}.csv"
            
            # Write particle data to CSV file
            with open(filename, 'w') as fout:
                # Write header
                fout.write("ID,X_Position,Y_Position,velocity_X,velocity_Y,acceleration_x,acceleration_y,Density,Pressure\n")
                
                # Write particle data
                for particle in self.particlePositions:
                    if particle.id < SPH.countFluid:
                        fout.write(f"{particle.id},{particle.x},{particle.y},{particle.u},{particle.v},{particle.acceleration_x},{particle.acceleration_y},{particle.rho},{particle.pressure}\n")
        
        if SPH.sharedStaticData % self.print_time == 0:
            # Create debugging output
            with open("print.txt", 'w') as fout:
                fout.write(f"sharedStaticData={SPH.sharedStaticData}\n")
                fout.write(f"静水算例 2025.5.8时间步长{SPH.sharedStaticData}\n")
                
                # Write x positions for first 40 particles
                fout.write("X_Position\n")
                for i in range(min(40, 51)):
                    fout.write(f"{self.particlePositions[i].x},")
                fout.write("\n\n")
                
                # Write y positions for first 40 particles
                fout.write("Y_Position\n")
                for i in range(min(40, 51)):
                    fout.write(f"{self.particlePositions[i].y},")
                fout.write("\n\n")
                
                # Write x velocities for first 40 particles
                fout.write("VX\n")
                for i in range(min(40, 51)):
                    fout.write(f"{self.particlePositions[i].u},")
                fout.write("\n\n")
                
                # Write x positions for all particles
                fout.write("X_Position\n")
                for i in range(len(self.particlePositions)):
                    fout.write(f"{self.particlePositions[i].x},")
                fout.write("\n\n")
                
                # Write y positions for all particles
                fout.write("Y_Position\n")
                for i in range(len(self.particlePositions)):
                    fout.write(f"{self.particlePositions[i].y},")
                fout.write("\n\n")
                
                # Write x velocities for all particles
                fout.write("VX\n")
                for i in range(len(self.particlePositions)):
                    fout.write(f"{self.particlePositions[i].u}, ")
                fout.write("\n\n")
                
                # Write pressures for all particles
                fout.write("PRESSURE\n")
                for i in range(len(self.particlePositions)):
                    fout.write(f"{self.particlePositions[i].pressure}, ")
                fout.write("\n\n")

                # Write y positions for virtual particles
                fout.write("YPosition\n")
                for i in range(self.particle_number, len(self.particlePositions)):
                    fout.write(f"{self.particlePositions[i].y},")
                fout.write("\n\n")
                
                # Write pressures for fluid particles
                fout.write("PRESSURE\n")
                for i in range(self.particle_number):
                    fout.write(f"{self.particlePositions[i].pressure}, ")
                fout.write("\n\n")
                
                # Write pressures for virtual particles
                fout.write("PRESSURE_VIRYUAL\n")
                for i in range(self.particle_number, len(self.particlePositions)):
                    fout.write(f"{self.particlePositions[i].pressure}, ")
                fout.write("\n\n")
    
