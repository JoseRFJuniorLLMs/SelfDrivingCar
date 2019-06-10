# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---
## Introduction
In this project I have implemented PID controller to control the motion of a car in Udacity's simulator. PID controller is used to adjust the car position on road to the recieved values of speed and angle. The simulator calculates the cross-track-error(which is the error in car position w.r.t. the canter lane), speed and angle to the PID controller and the controller responds back with the appropriate steering angle and the throttle to drive the car.

### Describe the effect each of the P, I, D components had in your implementation.
- "P" corresponds to the proportional part of the controller. It tries to keep the car in the center of the road. It changes the steering angle proportionally accoring to the CTE.
- "I' corresponds to the integral part of the controller. It tries to eliminate a possible bias on the controlled system that could prevent the error to be eliminated. It seeks to eliminate the residual error by adding a control effect due to the historic cumulative value of the error.
- "D"  corresponds to the differential part of controller. It helps to counteract the proportional trend to overshoot the center line by smoothing the approach to it. It also checks if a car has mantained an error over a period of time (which might be due wrong alignment of wheel) and tries to minimize this. 
### Describe how the final hyperparameters were chosen
I arrived at the final values of hyperparameters by manual trial and error approach. Initially I started with same values of Kp, Kd and Ki as were given in Udacity's lesson, but the car was oscillating too much using those values. Then I tried with Kd of 5.0 which allowed the car to have a term of differential error according to the change in CTE. Also, I kept the integral coefficient small enough to smoothen the adjustment of the car to the desired values. 
After a lot of trial and runs my car was able to drive the complete track with below mentioned values of hyper-parameters.
##### P: 0.15
##### I: 0.00005
##### D: 5

## Simulation

### The vehicle must successfully drive a lap around the track.
The car successfully drived a lap without leaving the track. There are some places where the car is osscilating but it successfully recovers itself and tries to keep itself in the middle of the road.
[Video Link](https://www.youtube.com/watch?v=rzpKJyElUUk)
