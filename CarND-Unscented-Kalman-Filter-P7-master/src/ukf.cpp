#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 6;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.7;

	//DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;
	//DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

	/**
	TODO:
	Complete the initialization. See ukf.h for other member properties.
	Hint: one or more values initialized above might be wildly off...
	*/

	n_x_ = 5;
	lambda_ = 3 - n_x_;
	
	n_aug_ = 7;
	weights_ = VectorXd(2 * n_aug_ + 1);
	
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
	
	is_initialized_ = false;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_){

		x_ << 1, 1, 0, 0, 0;

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
			/**
			Converting from polar to cartesian coordinates and updating state vector.
			*/
			x_(0) = meas_package.raw_measurements_(0) * cos(meas_package.raw_measurements_(1));
			x_(1) = meas_package.raw_measurements_(0) * sin(meas_package.raw_measurements_(1));
		}else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
			x_(0) = meas_package.raw_measurements_(0);
			x_(1) = meas_package.raw_measurements_(1);
		}

		P_ << 1, 0, 0, 0, 0,
			  0, 1, 0, 0, 0,
			  0, 0, 1, 0, 0,
			  0, 0, 0, 1, 0,
			  0, 0, 0, 0, 1;

		time_us_ = meas_package.timestamp_;

		// done initializing, no need to predict or update
		is_initialized_ = true;

		return;
	}

	/*****************************************************************************
	*  Prediction
	****************************************************************************/

	double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;  

	Prediction(delta_t);

	cout << "#############" << x_ << endl;

	/*****************************************************************************
	*  Update
	****************************************************************************/

	if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) & (use_radar_ == true)) {
		UpdateRadar(meas_package);
	} else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) & (use_laser_ == true)) {
		UpdateLidar(meas_package);
	}

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	/**
	TODO:
	Complete this function! Estimate the object's location. Modify the state
	vector, x_. Predict sigma points, the state, and the state covariance matrix.
	*/

	VectorXd x_aug = VectorXd(7);
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	MatrixXd P_aug = MatrixXd(7, 7);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5,5) = P_;
	P_aug(5,5) = std_a_ * std_a_;
	P_aug(6,6) = std_yawdd_ * std_yawdd_;
	
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	Xsig_aug.col(0)  = x_aug;
	MatrixXd L = P_aug.llt().matrixL();
	
	for (int i = 0; i< n_aug_; i++)
	{
		Xsig_aug.col(i+1)	=	x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i+1+n_aug_)	=	x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}

	//code to predict sigma points
	for (int i = 0; i< 2*n_aug_+1; i++){
		
		double p_x = Xsig_aug(0,i);
		double p_y = Xsig_aug(1,i);
		double v = Xsig_aug(2,i);
		double yaw = Xsig_aug(3,i);
		double yawd = Xsig_aug(4,i);
		double nu_a = Xsig_aug(5,i);
		double nu_yawdd = Xsig_aug(6,i);
		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;
		
		//state values
		double px_p, py_p;

		//code to avoid division by zero exception
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
			py_p = p_y + v / yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
		}else {
			px_p = p_x + v * delta_t * cos(yaw);
			py_p = p_y + v * delta_t * sin(yaw);
		}

		//noise
		px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
		py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
		yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
		yawd_p = yawd_p + nu_yawdd * delta_t;
		v_p = v_p + nu_a * delta_t;

		//writing predicted sigma points into right column
		Xsig_pred_(0,i) = px_p;
		Xsig_pred_(1,i) = py_p;
		Xsig_pred_(2,i) = v_p;
		Xsig_pred_(3,i) = yaw_p;
		Xsig_pred_(4,i) = yawd_p;
	}

	cout << Xsig_pred_ << endl;

	weights_(0) = lambda_ / (lambda_ + n_aug_);
	
	for (int i=1; i < (2 * n_aug_ + 1); i++) {  
		weights_(i) = 0.5 / (n_aug_+lambda_);
	}

	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
		
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.* M_PI;

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the lidar NIS.
  */
	MatrixXd H_ = MatrixXd(2, 5);
	H_ << 1, 0, 0, 0, 0, 
		0, 1, 0, 0, 0;

	MatrixXd z = VectorXd(n_x_);
	z = meas_package.raw_measurements_;

	VectorXd y = z - (H_ * x_);
	MatrixXd Ht = H_.transpose();

	MatrixXd R_laser_ = MatrixXd(2, 2);
	R_laser_ << std_laspx_ * std_laspx_, 0,
				0, std_laspy_ * std_laspy_;

	MatrixXd S = H_ * P_ * Ht + R_laser_;
	MatrixXd Si = S.inverse();
	
	MatrixXd K = P_ * Ht * Si;

	// new state
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	x_ = x_ + (K * y);
	P_ = (I - K * H_) * P_;

	double NIS_laser_ = 0.0;
	NIS_laser_ = y.transpose() * S.inverse() * y;
	cout << "NIS Lidar ---> "<< NIS_laser_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the radar NIS.
  */
  
	int n_z = 3;

	MatrixXd z = VectorXd(n_x_);
	z = meas_package.raw_measurements_;

	MatrixXd z_sig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//transforming sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v  = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);
		double v1 = cos(yaw) * v;
		double v2 = sin(yaw) * v;

		z_sig(0,i) = sqrt(p_x * p_x + p_y * p_y);         
		z_sig(1,i) = atan2(p_y, p_x);               
		z_sig(2,i) = (p_x * v1 + p_y * v2 ) / sqrt(p_x * p_x + p_y * p_y);
	}

	//mean of predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * z_sig.col(i);
	}

	MatrixXd S = MatrixXd(n_z,n_z);
	S.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; i++){
		VectorXd z_diff = z_sig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	MatrixXd R_radar_ = MatrixXd(n_z,n_z);
	R_radar_ <<    std_radr_*std_radr_, 0, 0,
				0, std_radphi_*std_radphi_, 0,
				0, 0,std_radrd_*std_radrd_;
	S = S + R_radar_;

	//cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);
	
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3) >  M_PI) x_diff(3) -= 2.* M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2.* M_PI;

		
		VectorXd z_diff = z_sig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
		while (z_diff(1) <-M_PI) z_diff(1) += 2. * M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//kalman gain
	MatrixXd K = Tc * S.inverse();

	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1) -= 2. * M_PI;
	while (z_diff(1)<-M_PI) z_diff(1) += 2. * M_PI;

	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	double NIS_radar_ = 0.0;
	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
	cout << "NIS radar: "<< NIS_radar_<< endl; 
}