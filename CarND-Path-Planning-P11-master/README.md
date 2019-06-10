# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program

## Rubrics

### Compilation
Code must compile without errors with `cmake` and make.

### Valid Trajectories
-   **The code compiles correctly :-** The code compiled successfully. I used `spline` for smoothening of the generated way points and the `spline.h` is included in `src` folder of the project directory.
-   **The car is able to drive at least 4.32 miles without incident :-** The car is able to drive across the highway easily. I tested it for 12 miles.
-   **The car drives according to the speed limit :-** The car drove within the speed limit of around 49.5 mph. 
-  **Max Acceleration and Jerk are not Exceeded :-** There was not an incident in entire 12 miles that I watched. 
-   **Car does not have collisions :-** There was not an incident in entire 12 miles that I watched.
-   **The car stays in its lane, except for the time between changing lanes :-** The car stays inside the lane for whole time except for the time between changing lanes. 
-   **The car is able to change lanes smoothly :-** During lane change the car reduces its speed to make a smooth transition from source to destination lane.

### Reflection

#### Prediction
The simulator sends a number of things like Car's location, velocity, yaw rate, speed, `frenet` coordinates and sensor fusion data. By using these values we predict the behavior of other objects in our case it is vehicles in future and we plan behavior of our car based on the behavior of the nearby objects. To fit the polynomial(path coordinates) in the implementation of this project I used `splin` library. We learned about jerk minimization by using polynomial fitting in our classroom but `splin.h` fits polynomial smoothly and it is well-proven library so I preferred to use this.

#### Behavior Planning
I have implemented behavior planning in following steps:-

1.  Check if there is any car in our path or not in next 30 mtrs.
2.  If a car is present 30 m ahead and it is slow moving try to change the lane.
3.  For changing lane first check for both the side lanes(if the car is in middle lane) for if there is no vehicle in near proximity from a current location of the car(i.e 30 m ahead and 15 m behind the car).
4.  If there is no vehicle in next 30 mtrs in both left and right lane, so in that case we see the speed of vehicles on these lane and based on that we decide which lane to choose. Here I am choosing the fast moving lane as it is the better option and less prone to accident.
5.  If no option for lane change then reduce the speed of the car to avoid collision.

#### Trajectory Generation
To compute the trajectory we use car speed, the speed of the surrounding cars, current lane, intended lane and previous points. For making trajectory smoother we add immediate two points from the previous trajectory. If there are no previous points then we use yaw rate and current car coordinates to compute previous points. After this we add evenly 30m spaced points ahead of the starting reference in frenet coordinates. To make mathematics bit easier we shift all the points car coordinate system and after processing, we convert them back to map coordinate system before passing those to the car.
```
vector<double> next_wp0 =  getXY(car_s+30,(2+4*lane),map_waypoints_s, map_waypoints_x,map_waypoints_y);
vector<double> next_wp1 =  getXY(car_s+60,(2+4*lane),map_waypoints_s, map_waypoints_x,map_waypoints_y);
vector<double> next_wp2 =  getXY(car_s+90,(2+4*lane),map_waypoints_s, map_waypoints_x,map_waypoints_y);
```

We add these points to the list of x and y points. After this, we transform the coordinates such that our car is at (0,0) with a heading of 0 degrees.

```
for(int i=0; i<ptsx.size(); i++){
	double shift_x = ptsx[i] - ref_x;
	double shift_y = ptsy[i] - ref_y;
	
	ptsx[i] = (shift_x*cos(0-ref_yaw)-shift_y*sin(0-ref_yaw));
	ptsy[i] = (shift_x*sin(0-ref_yaw)+ shift_y*cos(0-ref_yaw));
}
```

Then we fit a spline onto these points and convert the points that are lying on the spline as the points the car should follow.

Code to switch lanes:
``` 
if(too_close ==  true){
	ref_vel -= .224;
	if(left_lane_occupied!=true  && right_lane_occupied!=true  && lane==1){
		if(left_lane_vehicle_speed>right_lane_vehicle_speed){
			lane-=1;
		}else{
			lane+=1;
		}
	}else  if(left_lane_occupied!=true  && lane>0){
		// switch to left lane
		lane-=1;
	}else  if(right_lane_occupied !=  true  && lane<2){
		// switch to right lane
		lane+=1;
	}
}else  if(ref_vel<49.5){
	ref_vel+=.224;
}
```

   
### Simulator.
You can download the Term3 Simulator which contains the Path Planning Project from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).

### Goals
In this project your goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. You will be provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

#### The map of the highway is in data/highway_map.txt
Each waypoint in the list contains  [x,y,s,dx,dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet s value, distance along the road, goes from 0 to 6945.554.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

## Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.

## Tips

A really helpful resource for doing this project and creating smooth trajectories was using http://kluge.in-chemnitz.de/opensource/spline/, the spline function is in a single hearder file is really easy to use.

---

## Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!


## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

