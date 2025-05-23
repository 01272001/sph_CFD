from hydrostatic_vlx import SPH

def main():
    example = SPH()
    example.creatparticle()  
    # initialize
    SPH.sharedStaticData = 0    
    total_time = 0.0    
    while SPH.sharedStaticData < 100000:  # max step
        # 动态计算时间步长
        deltaTime = 0.00005  

        example.resetAcceleration()      
        # grid and nnps   
        example.grids()
        example.Virtual_particle()
        example.InteractingParticle() 
        example.divergence_distance()             
        
        example.Density(deltaTime)
        example.Pressure()
        example.mirror()        
        # acceleration
        example.visc()
        example.PressureGradient()       
        
        example.update(deltaTime,total_time)        
        # erease virtual particle
        example.erase()       
        # output
        example.print()        
 
        total_time += deltaTime
if __name__ == "__main__":
    main()