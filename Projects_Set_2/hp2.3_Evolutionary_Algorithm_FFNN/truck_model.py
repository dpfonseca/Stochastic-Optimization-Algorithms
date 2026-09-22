import math
from slopes import get_slope_angle
import numpy as np


# All units are in SI
GRAVITY = 9.8 

class truck_model_class:

    def __init__(self, maximum_temperature, mass, constant_tau, constant_c_h, ambient_temperature, constant_c_b):
        
        self.maximum_temperature = maximum_temperature
        self.mass = mass
        self.constant_tau = constant_tau
        self.constant_c_h = constant_c_h
        self.ambient_temperature = ambient_temperature
        self.constant_c_b = constant_c_b

        self.position, self.velocity, self.angle = 0.0, 20.0, 0.0
        self.temperature = 500
        self.gear = 7
        self.pedal_pressure = 0.0

    def force_of_gravity(self):
        force_gravity = self.mass * GRAVITY * math.sin(np.deg2rad(self.angle))
        return force_gravity
    

    def force_of_foundational_brakes(self):
        force_foundational_breaks = ( (self.mass * GRAVITY)/20 ) * self.pedal_pressure

        if self.temperature >= (self.maximum_temperature - 100):
            force_foundational_breaks *= math.exp( -(self.temperature - (self.maximum_temperature - 100))/100 )
        
        return force_foundational_breaks
    

    def temperature_change(self, time_step_size):
        
        relative_temperature = self.temperature - self.ambient_temperature

        if self.pedal_pressure < 0.01:
            relative_temperature = - time_step_size * (relative_temperature/self.constant_tau)

        else:
            relative_temperature = time_step_size * self.constant_c_h * self.pedal_pressure
        
        return relative_temperature
    

    def force_of_engine_breaks(self):
    
        force_engine_breaks = self.constant_c_b

        if self.gear == 1:
            force_engine_breaks *= 7.0
        if self.gear == 2: 
            force_engine_breaks *= 5.0
        if self.gear == 3: 
            force_engine_breaks *= 4.0
        if self.gear == 4: 
            force_engine_breaks *= 3.0
        if self.gear == 5: 
            force_engine_breaks *= 2.5
        if self.gear == 6: 
            force_engine_breaks *= 2.0
        if self.gear == 7: 
            force_engine_breaks *= 1.6
        if self.gear == 8: 
            force_engine_breaks *= 1.4
        if self.gear == 9: 
            force_engine_breaks *= 1.2
        if self.gear == 10: 
            force_engine_breaks *= 1
        
        return force_engine_breaks
    
    def velocity_change(self, time_step_size, previous_velocity):
        
        derivative_velocity = (1/self.mass) * (self.force_of_gravity() - self.force_of_foundational_brakes() - self.force_of_engine_breaks())

        new_velocity = (derivative_velocity * time_step_size) + previous_velocity

        return new_velocity
    
        
