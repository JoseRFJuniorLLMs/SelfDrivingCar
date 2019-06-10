#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    // Initializing the PID coefficients
    this -> Kp = Kp;
    this -> Ki = Ki;
    this -> Kd = Kd;

    // Inirializing PID coefficients errors
    this -> p_error = 0;
    this -> i_error = 0;
    this -> d_error = 0;
}

void PID::UpdateError(double cte) {
  d_error = cte - p_error;
  p_error = cte;
  i_error += cte;
}


double PID::TotalError() {
    // Return total error
    return -(Kp * p_error + Ki * i_error + Kd * d_error); 
}

