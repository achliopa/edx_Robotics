# edX Course: ColumbiaX: CSMM.103x Robotics

* [Course Link](https://www.edx.org/course/robotics-2)
* [Course Repo]()

## Robotics Course: Getting Started

### Recommended Readings

* Lorenzo Sciavicco and Bruno Siciliano, Modelling and control of robot manipulators, Springer
* Saeed B. Niku, Introduction to Robotics, Wiley
* Mark W. Spong, Seth Hutchinson and M. Vidyasagar, Robot Modeling and Control, Wiley
* Roland Siegwart, Introduction to Autonomous Mobile Robots, MIT Press.
* Peter Corke, Robotics, vision and control : fundamental algorithms in MATLAB, Springer

## Self-Assessment Activity 


### Jacobians

* This section is designed to introduce the Jacobian matrix using a simple example that resembles more difficult problems you will see later in the course.
* This assessment tests the following skills:
    * Making a function in Python
    * Building a matrix in Python
    * Partial differentiation
* Problem: Given the following system of equations: Define a Jacobian matrix with partial derivatives of functions x and y with respect to a and b

> x = A*cos(a) + B*cos(a+b)

> y = A*sin(a) + B*sin(a+b)


|   J = |       |
|  ---  |  ---  |
| dx/da | dx/db |
| dy/da | dy/db |

* we calculate the derivatives

> x = A*cos(a) + B*cos(a+b) = A*cos(a) + B*cos(a)*cos(b) - B*sin(a)*sin(b) 

> y = A*sin(a) + B*sin(a+b) = A*sin(a) + B*sin(a)*cos(b) + B*cos(a)*sin(b)

> dx/da = - A*sin(a) - B*sin(a)*cos(b) - B*cos(a)*sin(b) = - A*sin(a) - B*sin(a+b) = -y

> dy/da = A*cos(a) + B*cos(a)*cos(b) - B*sin(a)*sin(b) = A*cos(a) + B*cos(a+b) = x

> dx/db = A*cos(a) - B*cos(a)*sin(b) - B*sin(a)*cos(b) = A*cos(a) - B*sin(a+b)
  
> dy/db = A*sin(a) - B*sin(a)*sin(b) + B*cos(a)*cos(b) = A*sin(a) + B*cos(a+b)

```
import numpy as np

# Coefficients - in meters
A = 0.7
B = 0.3

# Angles - in degrees
a = 45
b = 60

# x = A*cos(a) + B*cos(a+b)
# y = A*sin(a) + B*sin(a+b)

# J = [dx/da, dx/db
#      dy/da, dy/db]


def compute_jacobian():
    J = np.ndarray((2,2))
    ###
    ### YOUR CODE HERE
    J[0,0] = - A*np.sin(a*np.pi/180) - B*np.sin((a+b)*np.pi/180)
    J[0,1] = A*np.cos(a*np.pi/180) - B*np.sin((a+b)*np.pi/180)
    J[1,0] = A*np.cos(a*np.pi/180) + B*np.cos((a+b)*np.pi/180)
    J[1,1] = A*np.sin(a*np.pi/180) + B*np.cos((a+b)*np.pi/180)
    ###
    return J

J = compute_jacobian()
print (J)
```

### Trigonometric Functions

* In this section, you are going to find the angle between two vectors with the help of trigonometric functions.
* Vector a = [1, 0] and b = [x, y] are in the same plane. Try to get the angle(-180, 180] between these two vectors
    * Use math.degrees() function to convert radian to angle.
    * Please choose the appropriate function from the following ones: math.sin(), math.asin(), math.cos(), math.acos(), math.tan(), math.atan(), math.atan2(), math.fabs().
```
import math
import numpy as np
# Complete the function below

def angle2D(x, y):
    ###
    ### YOUR CODE HERE
    a = np.array([1,0])
    b = np.array([x,y])
    angle = math.degrees(math.acos(((a[0]*b[0])+(a[1]*b[1]))/(np.linalg.norm(a)*np.linalg.norm(b))))
    ###
    return angle
# Run this cell to generate vector b

b = np.random.randint(-10, 10, 2)
x = b[0]
y = b[1]

if x == 0 and y == 0:
    x = 1
    y = 1

angle1 = angle2D(x, y)
```

* Now, you are going to get the angle between two vectors c = [c0, c1, c2] and d = [d0, d1, d2] in the same 3D space
    * Use math.degrees() function to convert radian to angle.
    * Please choose the appropriate function from the following ones: math.sin(), math.asin(), math.cos(), math.acos(), math.tan(), math.atan(), math.atan2(), math.fabs().
```
# Complete the function below

def angle3D(c, d):
    ###
    angle = math.degrees(math.acos(np.dot(c,d)/(np.linalg.norm(c)*np.linalg.norm(d))))
    ###
    return angle
# Run this cell to generate vector c and d

c = np.random.randint(-10, 10, 3)
d = np.random.randint(-10, 10, 3)

if np.linalg.norm(c) == 0:
    c = np.array([1,1,1])
if np.linalg.norm(d) == 0:
    d = np.array([-1,-1,-1])

angle2 = angle3D(c, d)
```

### Project 0 (Ungraded)

* This assignment is meant to make sure that you are familiar with the most basic functions of ROS. It is an ungraded project that all students can work on. We encourage students who have not yet signed up for the verified track to try this project and get a sense of the type of projects you will see in this course.
* In this assignment you must write a publisher node that publishes a message. You will publish a single value of type String from std_msgs that contains the message "robotics is fun" to the topic called 'quotes'. The grader will be subscribed to this topic and receive the message that you publish.
* Setup
    * Access your Vocareum workspace for Project 0
    * Start by running `source setup_project0.sh`  in the command line terminal. You should do this first every time you load or reload your workspace. You must run this command before trying to invoke any ROS commands (catkin_make, roscd, etc.). This will also start a roscore for your session. Please do not start your own roscore.
* Implementation
    * You must implement your code in the file `~/catkin_ws/src/project0_solution/scripts/solution.py` . This file has already been created for you and any starter code has been placed inside. 
* Testing
    * To test your code, you have multiple options:
    * Add some debug output to your publisher (i.e. print the number you have just received every time you are about to publish). 
      Then simply run your node `rosrun project0_solution solution.py`. This is useful to see that you are getting to the right place in your code, but will not tell if you are actually publishing, and publishing to the right topic.
    * Simply submit your code. Your code will be automatically graded and after a short while you should find a 'Submission Report' under the 'Details' tab which will contain output from the grading script.
    * (Requires more advanced Linux skills) Run your node in the background `rosrun project0_solution solution.py &`. 
      This frees up the console so you can manually subscribe to the quotes topic `rostopic echo quotes` and see that something is indeed being published. 
      Of course, you'll then need to manually kill your node, by retrieving the right process number and sending it a SIGINT signal using the `kill <PID>` command.
* solution: the publisher node sending message to topic every 2 sec
```
#!/usr/bin/env python  
import rospy

from std_msgs.msg import String


def talker():
    rospy.init_node('project0_solution', anonymous=True)
    pub = rospy.Publisher('/quotes', String, queue_size=10)
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        msg = String()
        msg.data = 'robotics is fun'
        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
```

### Matrix Operation

* This is an entry-level self-assessment of matrix operation. In this section, you are required to complete three functions:
    * Vector dot product
    * Matrix dot product
    * Matrix transpose operation
* These sections only involve some basic linear algebra knowledge, so we strongly suggest you write those functions by hand instead of using library.
* Vector dot product: a is a 1X3 vector and b is a 3X1 vector. Please fill in your solution below.
```
# Complete the function below
def VectorDotProduct(a, b):
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]
    b1 = b[0][0]
    b2 = b[1][0]
    b3 = b[2][0]
    ###
    vector_result = a1*b1+a2*b2+a3*b3
    ###
    return vector_result
a = [1,2,3]
b = [[1],
     [2],
     [3]]
vector_result = VectorDotProduct(a, b)
print(vector_result)
```
* Matrix dot product:
    * a and b are both 3X3 matrix. Please fill in your solution below.
    * Hint: You can use the function: VectorDotProduct(a,b) at here
```
# Complete the function below
def MatrixDotProduct(a, b):
    a_row_0 = a[0]
    a_row_1 = a[1]
    a_row_2 = a[2]
    b_col_0 = [[b[i][0]] for i in range(3)]
    b_col_1 = [[b[i][1]] for i in range(3)]
    b_col_2 = [[b[i][2]] for i in range(3)]
    ###
    matrix_result = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range (3):
        for j in range (3):
            matrix_result[i][j] = VectorDotProduct(a[i], [[b[k][j]] for k in range(3)])
    ###
    return matrix_result
a = [[1,2,3],
     [4,5,6],
     [7,8,9]]
b = [[1,2,3],
     [4,5,6],
     [7,8,9]]
matrix_result = MatrixDotProduct(a, b)
print(matrix_result
```
* Matrix transpose: In this time, a is a 3X3 matrix. Please fill in your solution below
```
# Complete the function below
def MatrixTranspose(a):
    ###
    matrix_result = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(3):
        for j in range(3):
            matrix_result[i][j] = a[j][i]
    ###
    return matrix_result
a = [[1,2,3],
     [4,5,6],
     [7,8,9]]

matrix_result = MatrixTranspose(a)
print(matrix_result)
```

## Week 1: Introduction to Robotics

### 1.1 What Is a Robot?

* Robot Companies
    * KUKA
    * Kiva Systems (Amazon Robot)
    * AETHEON
    * Savioke
    * Mayfield Robotics
    * Boston Dynamics(BigDog)
    * GoogleX (project Wing)
* Manipulation
    * Robot Arm: manipulation
        * industrial maipulation: manufacturing, assembly lines..
        * pre programmed by a human operator on the movement to perform
        * repeatable trajectory
        * precision
        * strength
    * Robotic Surgery
        * teleoperation
    * Explosive Opbject Disposal (EOD) Robot: teleoperaton
    * General Purpose manipulator (Autonomous) 
        * Learning (DNN)
        * Sensing
        * Motion Planning
* Mobility
    * wheeled robots
        * more domains each day: warehouses, hospitals, hotels, home
        * human environment is not built for robots
        * unstructured env
    * autonomous cars
        * semantic perception (what is what)
        * lidars, camera, sensors are critical
        * it must respond fast to conditions
        * collision avoidance
    * legged automotion
        * dynamics
        * legged + wheeled
    * mobility + manipulation: disaster reponse
    * aerial robots
        * fixed wing 
    * underwater
* Smart sensors
    * Leaning Thermostat from Nest (learns from habits)
* Possible Definition for a Robot:
    * A device that can sense, plan and act
    * A self-powered device that effects physical change to the world
* Robotics: SW and HW (mind and body)

### 1.2 Robotics and AI - at the Beginning

* Robotics and AI:
* pre-programmed robotic amnipulators have had a profound effect in society
* New frontier: unstructured environments. programmer cannot provide exact instructions in advance for every possible scenario the robot will see.
* Critical Abilities
    * Sensing
    * Reacting
    * Planning
* Applies to manipulation and mobility
    * Classic manipulators enhanced with sensing
    * Manipulators safe to work with
* Do we trust the robot to be intelligent enough to do tasks on ts own?
* (Artificial) Intelligence: also no universally accepted definition
    * the ability to react appropriately when faced with unforeseen situations (applies to humans and robots)
    * sensing and planning is the key

### 1.3 What we will cover in this course

* The foundation for intelligent robots (manipulation focus)
    * 3D Space and transforms
    * Manipulation: how to model and manipulate robot arms
    * mobility: how to model and analyze mobile robots
    * motion planning
    * assignments using the open-source Robot Operating System (ROS)
* Not covered in class:
    * Building Robot Hardware (design, mechatronics)
    * Embedded programming
    * Sensing and perception

### 1.4 Introduction to ROS

* A colection of libraries, tools and conventions
    * Plumbing
        * code organization and management
        * communication between components
    * Tools
        * introspection
        * visualization
    * Capabilities
        * navigation (localization, path planning, etc)
        * perception (object recognition, etc)
        * manipulation (arm motion planning etc)
        * ....
    * Ecosystem
        * users
        * support forum
        * conferences
* Runs on top of OS (Ubuntu Linux)
* it organizes code in packages and nodes
    * laser driver node (outputs laser data to localization)
    * camera driver node (outputs image to machine vision node)
    * machine vision node (receicee image)(outputs image metadata to localization)
    * localization node (outputs robot location to path planner)
    * path planner node (receives the goal)(outputs motor commands)
    * motor friver nodes
* it has a workspace for our project
* it has a visualizer (rviz)
* ROS uses the publish subscribe mechanism for nodes to talk to each other through cannels called topics
* nodes can run on different machines and can be written in diferrent languages
* ROS is widely used in robotics research (especialy in academia) and increasingly by companies prototyping new cutting-edge ideas

### 1.5 ROS Use in this Course

* 2 ways to use ROS
* 1) Install it on our machine
    * We need an Ubuntu Linux machine
    * Full access to complete ROS ecosystem
* 2) Use the browser based interface
    * at least to do the assignements
* Wont teach ROS (learn on our own)
* Go to [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials)
    * Begginner Tutorials 1-6 and 11-13
* ROS natively supports 2 langs: Python and C++
* To follow up the course we must be able to
    * setup a catkin workspace
    * create apackage inside the workspace
    * (if using C++) compile the code in the package
    * publish to a topic subscribe to a topic

### Project 1

* This assignment is meant to make sure that you are familiar with the most basic functions of ROS. Please make sure that you have completed (or at least read through) the tutorials 1-6 & 11-13.
* In this assignment you are tasked with writing a node that subscribes to a topic and publishes to another. Your code will subscribe to a topic called 'two_ints', on which a custom message containing two integers can be broadcast. Make sure to familiarize yourself with the message format of this topic (have a look at the TwoInts.msg in the msg directory). Those two integers are to be added and the result published to topic 'sum' as an Int16 from std_msgs.
* Setup
    * Access your Vocareum workspace for Project 1
    * Start by running source setup_project1.sh  in the command line terminal. You should do this first every time you load or reload your workspace. You must run this command before trying to invoke any ROS commands (catkin_make, roscd, etc.). This will also start a roscore for your session. Please do not start your own roscore.
    * Once you have sourced this script, there will be a ROS package publishing random integers to the 'two_ints' topic every two seconds. 
* Implementation
    * You must implement your code in the file ~/catkin_ws/src/project1_solution/scripts/solution.py . This file has already been created for you and any starter code has been placed inside. 
* Testing
    * Add some debug output to your publisher (i.e. print the two numbers you have just received as well as their sum to the console every time you are about to publish). Then simply run your node (rosrun project1_solution solution.py). This is useful to see that you are getting to the right place in your code, but will not tell if you are actually publishing, and publishing to the right topic.
    * Simply submit your code. Your code will be automatically graded and after a short while you should find a 'Submission Report' under the 'Details' tab which will contain output from the grading script.
    * (Requires more advanced Linux skills) Run your node in the background (rosrun project1_solution solution.py &). This frees up the console so you can manually subscribe to the sum topic (rostopic echo sum) and see that something is indeed being published. Of course, you'll then need to manually kill your node, by retrieving the right process number and sending it a SIGINT signal using the kill command.

## Week 2: Reasoning About Space and Transforms

### 2.1 Transforms Introduction

* robots are machines operating in physical space
* concepts on space apply on graphics
* we typically have a reference frame for the world (0,0,0)
* we assume we know where the target is in respcet to the refernce (coordinate) frame
* we also know where the robot arm is with respect to the coordinate frame
* the robot arm needs to know where the target is with respect to itself
* so we typically need to define where objects are with respect to each other
* so a mobile robot is not enough to know where the obstacles are in reference to a coordinate frame but relative to itself
* so in robotics we need to move between coordinate frames. this is called Transforms
* In 2D positional space: position of p is an 1x2 column vector with the projections on the 2 reference axes: 

> p = [[px],[py]] = [px,py]T (transpose)

* In 3D space respectively the position of p is an 1x3 column vector with projections on the 3 reference axes:

> p = [[px],[py],[pz]] = [px,py,pz]T

* There is no universal coordinate frame in robotics. there are numerous (for the robot, for the room
* In this course we will use capital letters to name the various coordinate frames.
* if we have p expressed in coordinate frame A our representation becomes

> <sup>A</sup>p = [[<sup>A</sup>px],[<sup>A</sup>py]] = [<sup>A</sup>px,<sup>A</sup>py]T (2D)
> <sup>A</sup>p = [[<sup>A</sup>px],[<sup>A</sup>py],[<sup>A</sup>pz]] = [<sup>A</sup>px,<sup>A</sup>py,<sup>A</sup>pz]T (3D)

* Linear algebra is our go to tool.
* if we have a camera at point B it will tell us where p is with respect to its own coordinate frame
* but we might need both p point vectors with respect to A and B. IF we know the *Transform* that gets us from coordinate frame A to coordinate frame B, and where point p is in coordinate frame B we will be able to compute the location of p in coordinate frame A.

> <sup>A</sup>T<sub>B</sub>.<sup>B</sup>p = <sup>A</sup>p

* we can chain transformations. if our point is observed by a camera in coordinate frame C. the camera is at the end of a robot that has at the basecoordinateframe B and the coordinate frame for the world is A. if we want to know the location of p with respect to A and we know the trasforms between coordinate frames and the position in reference to the camera C

> <sup>A</sup>T<sub>B</sub>.<sup>B</sup>T<sub>C</sub>.<sup>C</sup>p = <sup>A</sup>p

### 2.2 2D Rotations Part I 

* we will now try to define the transforms mathematically
* we assume 2 2D coordinate frames with same point of origin but rotated by an angle Θ
* we assume we know the point vector in referenc eto coordinate frame B: 
> <sup>B</sup>p = [<sup>B</sup>p<sub>x</sub>,<sup>B</sup>p<sub>y</sub>]
* we calculate the <sup>A</sup>p  considering transform
> <sup>A</sup>p<sub>x</sub> = <sup>B</sup>p<sub>x</sub>*cos(θ) - <sup>Β</sup>p<sub>y</sub>*sin(θ)
> <sup>A</sup>p<sub>y</sub> = <sup>B</sup>p<sub>x</sub>*sin(θ) + <sup>Β</sup>p<sub>y</sub>*cos(θ)
* we rewrite it in vector form
> <sup>A</sup>p = [[<sup>A</sup>p<sub>x</sub>],[<sup>A</sup>p<sub>y</sub>]] = [[cos(θ) , -sin(θ)],[sin(θ) , cos(θ)]] * [[<sup>Β</sup>p<sub>x</sub>],[<sup>Β</sup>p<sub>y</sub>]]
* the transofrmation of coordinate frames is a rotation matrix we can express as <sup>A</sup>R<sub>B</sub> as coordinates A and B differ only by rotation
* we give a numerical example. say the B  coordinate frame is A rotated by 45deg and we assume that point vector p reference to B is <sup>B</sup>p = [2,0]T
* we calculate point vector p in reference to A
> <sup>A</sup>p = <sup>A</sup>R<sub>B</sub> * <sup>B</sup>p = [[0.7,-0.7],[0.7,0.7]] * [[2],[0]] = [[1.4],[1.4]]
* We defined the Rotation matrix. in 2D we rotate accross the z axis which is invisible and points towards the viewer
* when we see a Rotation matrix the following hold true:
    * determinant of R is always 1 |R| = 1
    * Rotation matrices are always orthonormal (norm of every column and row is 1 , dot product of every 2 columns is 0, the dot product of every 2 row is 0)
    * R<sup>-1</sup> = R<sup>T</sup> is a useful rule as we can use the transpose of transform (rotate) matrix to go instead of A->B from B->A <sup>B</sup>R<sub>A</sub> = (<sup>A</sup>R<sub>B</sub>)<sup>T</sup>

### 2.3 2D Rotations Part II, 2D Translations

* we write again the rotation matrix
> ![p](https://latex.codecogs.com/gif.latex?%5Cfn_phv%20%5E%7BA%7DR_%7BB%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Ccos%28%5Ctheta%29%20%26%20-%5Csin%28%5Ctheta%29%5C%5C%20%5Csin%28%5Ctheta%29%20%26%20%5Ccos%28%5Ctheta%29%20%5Cend%7Bbmatrix%7D)
* the point p expressed in coordinate frame A can be represented as a vector matrix or a scalar multiplied by a unit axis (unit axis vectors)
> ![p](https://latex.codecogs.com/gif.latex?%5Cfn_phv%20p%20%3D%20%5Cbegin%7Bbmatrix%7D%20p_%7Bx%7D%20%5C%5C%20p_%7By%7D%20%5Cend%7Bbmatrix%7D%3Dp_%7Bx%7D%5Ccdot%5Cvec%7Bx%7D&plus;p_%7By%7D%5Ccdot%5Cvec%7By%7D%20%5C%3A%5C%3A%5C%3Awhere%5C%3A%5C%3A%5C%3A%5Cvec%7By%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%5C%3A%5C%3A%5C%3Aand%5C%3A%5C%3A%5C%3A%5Cvec%7By%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%5C%5C%201%20%5Cend%7Bbmatrix%7D)![p](https://latex.codecogs.com/gif.latex?%5Cfn_phv%20%5C%3A%5C%3A%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cvec%7Bx%7D%26%5Cvect%7By%7D%20%5Cend%7Bbmatrix%7D%5Ccdot%5Cbegin%7Bbmatrix%7D%20p_%7B%7Dx%5C%5Cp_%7By%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%5Ccdot%5Cbegin%7Bbmatrix%7D%20p_%7B%7Dx%5C%5Cp_%7By%7D%20%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%20p_%7B%7Dx%5C%5Cp_%7By%7D%20%5Cend%7Bbmatrix%7D)