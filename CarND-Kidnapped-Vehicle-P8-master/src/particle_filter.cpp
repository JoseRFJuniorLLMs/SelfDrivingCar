/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	default_random_engine gen;
	
	is_initialized = true;
	num_particles = 10;
	
	for(int i = 0; i < num_particles; i++){
		Particle p = Particle();
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(1);
	}


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
		
	for(int i=0; i < num_particles; i++){
		Particle &p = particles[i];

		// Predict px, py and theta 
		if(fabs(yaw_rate) > 0.00001){
			p.x = p.x + (velocity/ yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y = p.y + (velocity/ yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta = p.theta + yaw_rate * delta_t;
		} else{
			// No change in yaw rate
			p.x = p.x + velocity * cos(p.theta) * delta_t;
			p.y = p.y + velocity * sin(p.theta) * delta_t;
		}

		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		// Add noise to delta x, y and theta 
        p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		
		weights[i]=1;
	}




}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	int number_of_observations = observations.size();
	int number_of_landmarks	= map_landmarks.landmark_list.size();
	double gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
	
	double sig_x_2 = 2.0 * std_landmark[0] * std_landmark[0];
	double sig_y_2 = 2.0 * std_landmark[1] * std_landmark[1];

	for(int i = 0 ; i < num_particles; i++) {
		Particle &p = particles[i];
		double total_weight_of_current_particle = 1.0;
		
		for(int j=0; j < number_of_observations; j++){
			LandmarkObs obs = observations[j];
			// Transform observed coordinates to map coordinates
			double x_m = p.x + (obs.x * cos(p.theta))  - (obs.y * sin(p.theta));
			double y_m = p.y + (obs.x * sin(p.theta)) + (obs.y * cos(p.theta));
			double min_euc_dist = 9999.99;
			int nearest_landmark_pos = 0;

			for(int k=0; k<number_of_landmarks; k++){
				
				double x_f = map_landmarks.landmark_list[k].x_f;
				double y_f = map_landmarks.landmark_list[k].y_f;

				// Calculate euclidian dist between current landmark and transformed observed coordinate
				double euc_dist = dist(x_m, y_m, x_f, y_f);
				
				if(euc_dist < min_euc_dist){
					// Nearest neighbour
					min_euc_dist = euc_dist;
					nearest_landmark_pos = k;
				}
			}

			double x_square = (x_m - map_landmarks.landmark_list[nearest_landmark_pos].x_f) * 
							  (x_m - map_landmarks.landmark_list[nearest_landmark_pos].x_f);
			
			double y_square = (y_m - map_landmarks.landmark_list[nearest_landmark_pos].y_f) * 
							  (y_m - map_landmarks.landmark_list[nearest_landmark_pos].y_f);

			double exponent = (x_square / sig_x_2) + (y_square / sig_y_2);

			
			// calculate weight using normalization terms and exponent
			double current_obs_weight= gauss_norm * exp(-exponent);


			total_weight_of_current_particle *= current_obs_weight;
		}

		weights[i] = total_weight_of_current_particle;	
		p.weight = total_weight_of_current_particle;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> p;

    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> distribution(weights.begin(), weights.end());

    for(int i = 0; i < num_particles; i++){
        Particle particle = particles[distribution(gen)];
        p.push_back(particle);
    }
    particles = p;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
