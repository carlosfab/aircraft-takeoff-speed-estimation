# Aircraft Takeoff Speed Estimation

![Alt Text](images/poc.gif)

**UNDER CONSTRUCTION**

## About <a name = "about"></a>

This project aims to estimate the takeoff speed of aircraft using computer vision techniques. By analyzing video footage of aircraft takeoffs, the system detects and tracks the aircraft, calculates their speed, and provides real-time speed estimations in knots. The project utilizes the YOLOv8 object detection model and the Supervision library for video analysis and annotation.

The system can be useful for aviation enthusiasts, researchers, or professionals interested in monitoring and analyzing aircraft performance during takeoffs. It provides a non-intrusive way to estimate takeoff speeds and can potentially assist in identifying any unusual or abnormal takeoff patterns.

## Installation and Configuration

Here are the steps you need to follow to set up your development environment:

1. Clone the [Github Repository](https://github.com/carlosfab/aircraft-takeoff-speed-estimation.git) to your local machine and access the `aircraft-takeoff-speed-estimation` folder:

   ```bash
   git clone https://github.com/carlosfab/aircraft-takeoff-speed-estimation.git
   cd aircraft-takeoff-speed-estimation
   ```

2. Configure Poetry to create virtual environments within the project directory.

   ```bash
   poetry config virtualenvs.in-project true
   ```

3. Set up the `3.10.0` version of Python with Pyenv:

   ```bash
   pyenv install 3.10.0
   pyenv local 3.10.0
   ```

4. Install project dependencies:

   ```bash
   poetry install
   ```

5. Activate the virtual environment.

   ```bash
   poetry shell
   ```
