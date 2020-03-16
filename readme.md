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
> ![p](https://latex.codecogs.com/gif.latex?%5Cfn_phv%20p%20%3D%20%5Cbegin%7Bbmatrix%7D%20p_%7Bx%7D%20%5C%5C%20p_%7By%7D%20%5Cend%7Bbmatrix%7D%3Dp_%7Bx%7D%5Ccdot%5Cvec%7Bx%7D&plus;p_%7By%7D%5Ccdot%5Cvec%7By%7D%20%5C%3A%5C%3A%5C%3Awhere%5C%3A%5C%3A%5C%3A%5Cvec%7By%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%5C%3A%5C%3A%5C%3Aand%5C%3A%5C%3A%5C%3A%5Cvec%7By%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%5C%5C%201%20%5Cend%7Bbmatrix%7D)
* going further in the thought process
> ![p](https://latex.codecogs.com/gif.latex?%5Cfn_phv%20p%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cvec%7Bx%7D%26%5Cvect%7By%7D%20%5Cend%7Bbmatrix%7D%5Ccdot%5Cbegin%7Bbmatrix%7D%20p_%7B%7Dx%5C%5Cp_%7By%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%5Ccdot%5Cbegin%7Bbmatrix%7D%20p_%7B%7Dx%5C%5Cp_%7By%7D%20%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%20p_%7B%7Dx%5C%5Cp_%7By%7D%20%5Cend%7Bbmatrix%7D)
* if we want to exress p in reference to another coordinate frame which is rotated we just change unit axis vectors for the new frame
> ![p](https://latex.codecogs.com/gif.latex?%5Cfn_phv%20%5Cvec%7Bx%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Ccos%28%5Ctheta%29%5C%5C%5Csin%28%5Ctheta%29%20%5Cend%7Bbmatrix%7D%5C%3A%5C%3A%5C%3Aand%5C%3A%5C%3A%5C%3A%5Cvec%7By%7D%3D%5Cbegin%7Bbmatrix%7D%20-%5Csin%28%5Ctheta%29%5C%5C%5Ccos%28%5Ctheta%29%20%5Cend%7Bbmatrix%7D)
* if we put the new unit axis vectors in our equations we go back to the the point transorm to a new rotated frame using the rotation matrix going from frame A to a rotated frame B
* so the rotation matrix is actually the new unit axis vectors in column format (regarding the original coordinate frame)
* if i get a rotation matrix i can get the new coordinate frame if we extract the new unit vaxis vectors and represent them 
* then we can multiply vectors with a point in the rotated frame and get its position in the original frame
> ![p](https://latex.codecogs.com/gif.latex?%5E%7BA%7DR_%7BB%7D%3D%5Cbegin%7Bbmatrix%7D%20-0.7%20%26%200.7%5C%5C%20-0.7%20%26%20-0.7%20%5Cend%7Bbmatrix%7D%5C%3A%5C%3A%5C%3A%5E%7BB%7Dp%3D%5Cbegin%7Bbmatrix%7D2%5C%5C0%5Cend%7Bbmatrix%7D)
* the point represenation in coordinate frame A is:
> ![p](https://latex.codecogs.com/gif.latex?%5E%7BA%7Dp%20%3D%20%5E%7BA%7DR_%7BB%7D%5Ccdot%5E%7BB%7Dp%3D%5Cbegin%7Bbmatrix%7D%20-0.7%20%26%200.7%5C%5C%20-0.7%20%26%20-0.7%20%5Cend%7Bbmatrix%7D%5Ccdot%5Cbegin%7Bbmatrix%7D2%5C%5C0%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D-1.4%5C%5C-1.4%5Cend%7Bbmatrix%7D)
* the axis of a coordinate frame are always mutually orthogonal, perpendicular to each other (90deg) so the dot product of any axis columsn has to be zero
* Translations are simpler that rotations. translation is when 2 oordinate frames the axis have the same orientation bt the origin point is moved by tx and ty (translated)
* translation is expressed as
> ![p](https://latex.codecogs.com/gif.latex?%5E%7BA%7Dp%20%3D%20%5E%7BB%7Dp&plus;%5Cbegin%7Bbmatrix%7D%20t_%7Bx%7D%5C%5Ct_%7By%7D%20%5Cend%7Bbmatrix%7D)
* our goal is to combine translation and rotation which is the most general case in 2D also called Full Transform. it can be expressed as 
> ![p](https://latex.codecogs.com/gif.latex?%5E%7BA%7Dp%20%3D%20%5E%7BA%7DR_%7BB%7D%5Ccdot%5E%7BB%7Dp&plus;%5Cbegin%7Bbmatrix%7D%20t_%7Bx%7D%5C%5Ct_%7By%7D%20%5Cend%7Bbmatrix%7D)
* this is a 2step operation non convenient when we have to chain transformations

### 2.4 Homogenous Coordinates, 2D Transforms

* we can turn the 2 step operation of 2D transform to an 1 step operation refering to it using homogeneous coordinates where we express point in frame B as a 3 element column vector (the 1 is comonly used also in deep learning for constructing linear methods to express the addition)
> ![p](https://latex.codecogs.com/gif.latex?%5E%7BB%7Dp%20%3D%20%5Cbegin%7Bbmatrix%7Dp_%7Bx%7D%5C%5Cp_%7By%7D%5C%5C1%5Cend%7Bbmatrix%7D)
* the transformation matrix is expressed as 
> ![p](https://latex.codecogs.com/gif.latex?%5E%7BA%7DT_%7BB%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5E%7BA%7DR_%7BB%7D%20%26%20%5E%7BA%7Dt_%7BB%7D%5C%5C0%261%20%5Cend%7Bbmatrix%7D)
* we combine both to get an 1 step transform operation
> ![p](https://latex.codecogs.com/gif.latex?%5E%7BA%7Dp%3D%5Cbegin%7Bbmatrix%7D%5E%7BA%7Dp_%7Bx%7D%5C%5C%5E%7BA%7Dp_%7By%7D%5C%5C1%5Cend%7Bbmatrix%7D%3D%5E%7BA%7DT_%7BB%7D%5Ccdot%5E%7BB%7Dp%3D%5Cbegin%7Bbmatrix%7D%20%5E%7BA%7DR_%7BB%7D%20%26%20%5E%7BA%7Dt_%7BB%7D%5C%5C0%261%20%5Cend%7Bbmatrix%7D%5Ccdot%5Cbegin%7Bbmatrix%7D%5E%7BB%7Dp_%7Bx%7D%5C%5C%5E%7BB%7Dp_%7By%7D%5C%5C1%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%5Ccos%28%5Ctheta%29%26-%5Csin%28%5Ctheta%29%26t_%7Bx%7D%5C%5C%5Csin%28%5Ctheta%29%26%5Ccos%28%5Ctheta%29%26t_%7By%7D%5C%5C0%260%261%5Cend%7Bbmatrix%7D%5Ccdot%5Cbegin%7Bbmatrix%7D%5E%7BB%7Dp_%7Bx%7D%5C%5C%5E%7BB%7Dp_%7By%7D%5C%5C1%5Cend%7Bbmatrix%7D)
* Transformations are fundamental in Robotics it is the key to do any operation in 3D space and the form is always the one we used so far using the rotation matrix R and the translate vector t
* in graphics we can have other values instead of 0 and 1. in robotics its always 0 and 1

### 2.5 3D Transforms

* With 2D under out belt we go to 3D Transforms
* everything from 2D holds in 3D we just add the z coordinate
    * rotation matrix is 3x3
    * translation vector is 1x3
    * point vector in homgeneous coordinate form is 1x4
* in 2D rotation can be only along th Z axis. in 3D rotation can be on any axis
* rotation matrices have different form depending on which axis we are rotating on
* if we rotate across the x axis by θ
> ![x](https://latex.codecogs.com/gif.latex?R_%7Bx%7D%28%5Ctheta%29%3D%5Cbegin%7Bbmatrix%7D1%260%260%5C%5C0%26%5Ccos%28%5Ctheta%29%26-%5Csin%28%5Ctheta%29%5C%5C0%26%5Csin%28%5Ctheta%29%26%5Ccos%28%5Ctheta%29%5Cend%7Bbmatrix%7D)
* if we rotate across the y axis by θ
> ![y](https://latex.codecogs.com/gif.latex?R_%7By%7D%28%5Ctheta%29%3D%5Cbegin%7Bbmatrix%7D%5Ccos%28%5Ctheta%29%260%26%5Csin%28%5Ctheta%29%5C%5C0%261%260%5C%5C-%5Csin%28%5Ctheta%29%260%26%5Ccos%28%5Ctheta%29%5Cend%7Bbmatrix%7D)
* if we rotate across the z axis by θ
> ![z](https://latex.codecogs.com/gif.latex?R_%7Bz%7D%28%5Ctheta%29%3D%5Cbegin%7Bbmatrix%7D%5Ccos%28%5Ctheta%29%26-%5Csin%28%5Ctheta%29%260%5C%5C%5Csin%28%5Ctheta%29%26%5Ccos%28%5Ctheta%29%260%5C%5C0%260%261%5Cend%7Bbmatrix%7D)
* when we rotate on an axis this axis does not change. we can identify it in the rotation matrix as the axis column remains unchanged
* the structure for 3D transforms is the same as in 2D
> ![r](https://latex.codecogs.com/gif.latex?%5E%7BA%7DT_%7BB%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5E%7BA%7DR_%7BB%7D%20%26%20%5E%7BA%7Dt_%7BB%7D%5C%5C0%261%20%5Cend%7Bbmatrix%7D)
* the properties of the rotation matrix are the same in 3D as in 2D (orthonormal)
> ![ρ](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bvmatrix%7D%5E%7BA%7DR_%7BB%7D%5Cend%7Bvmatrix%7D%3D1%5C%3A%5C%3A%2C%5C%3A%5C%3AR%5E%7B-1%7D%3DR%5E%7BT%7D%5C%3A%5C%3A%2C%5C%3A%5C%3ARR%5E%7BT%7D%3DI)
* we present a valid 3D transform matrix with a rotation along the z axis
> ![t](https://latex.codecogs.com/gif.latex?%5E%7BA%7DT_%7BB%7D%3D%5Cbegin%7Bbmatrix%7D0%26-1%260%263%5C%5C1%260%260%263%5C%5C0%260%261%260%5C%5C0%260%260%261%5Cend%7Bbmatrix%7D)

### 2.6 Transforms: Different Perspectives

* say we observe from a train another in a platform and its moving backwards. how we are sure it is moving backwards and not our train moving forward
* when we have relative motion between 2 points it is maybe one is moving in one direction or the other on the opposite direction
* same holds for transforms
* if we consider the coordinate frame as the observation point or vantage point using transormation A->B is equivalent to moving the observation point from A to B keeping the point p stable
* or we can say that p is moving to a new position pNew using the Transformation matrix keeping the observation point / vantage point (coordinate frame) stable
* pNew is the same in both cases its just how we look at it. in 1st case observation point moves in the second the actual point. the Transformation matrix is the same so the movement of either point is the same.
* in a robotics application the question can be formulated as follows:
    * A camera mounted on a robot arm observes an object at position p relative to itself. The trasform from the world frame to the camera frame is T. what is the position of the object expressed in the world frame?
    * A mobile robot is at position p in the world frame. The robot moves by translation T expressed in the world. What is the new poisition  of the robot?
* in both cases we have: newP = T*p
* say we have 3 2D transforms and a point p
> ![p](https://latex.codecogs.com/gif.latex?T_%7B1%7D%3D%5Cbegin%7Bbmatrix%7D%201%260%262%5C%5C0%261%260%5C%5C0%260%261%5Cend%7Bbmatrix%7D%5C%3BT_%7B2%7D%3D%5Cbegin%7Bbmatrix%7D%20-1%260%260%5C%5C0%26-1%260%5C%5C0%260%261%5Cend%7Bbmatrix%7D%5C%3BT_%7B3%7D%3D%5Cbegin%7Bbmatrix%7D%201%260%260%5C%5C0%261%262%5C%5C0%260%261%5Cend%7Bbmatrix%7D%5C%3B%20point%5C%3Ap%3D%5Cbegin%7Bbmatrix%7D2%5C%5C2%5C%5C1%5Cend%7Bbmatrix%7D)
* what is the meaning of T3T2T1p is the point doing 3 moves its the vantage point or a combination?
* to chain the transforms when we tranform (move) the vantage point we go left to right starting from the identity frame and applying transforms T3->T2->T1 then we set the point position relative to the  final vantage point (coordinate frame)
* if we consider the point moving we apply the transforms right to left T1->T2->T3 on the point vector which represents the postitin to the original coordinate frame or identity frame. the vantage point does not change
* due to dimensions we can only left multiply the point vector with a transform matrix
* be careful: matrix multiplication is not commutative T0T1 != T1T0 
* order does matter when chaining transforms. always think on which coordinate frame we are at a given point when applying a new transform

### 2.7 Recap

* 2 problems that come up when moving to space
    * when my vantage point is changing the view of the world (perspective) changes
    * when the actual object moves
* In 3D space points are represented as 4 dimensonal vectors
> ![p](https://latex.codecogs.com/gif.latex?p%20%3D%20%5Cbegin%7Bbmatrix%7D%20p_%7Bx%7D%26p_%7By%7D%26p_%7Bz%7D%261%5Cend%7Bbmatrix%7D%5E%7BT%7D)
* A transform in 3D space has a specific form
> ![t](https://latex.codecogs.com/gif.latex?T%20%3D%20%5Cbegin%7Bbmatrix%7D%20R%26t%5C%5C0%261%5Cend%7Bbmatrix%7D)
* we can use a transform for 2 things:
    * move the vantage point using T1 then T2 then observe point p
    > ![t](https://latex.codecogs.com/gif.latex?T_%7B1%7D%20%5Ccdot%20T_%7B2%7D%20%5Ccdot%20p)
    * another way to express the problem is using the coordinate fram notation
    > ![l](https://latex.codecogs.com/gif.latex?%5E%7BA%7Dp%20%3D%20%5E%7BA%7DT_%7BB%7D%20%5Ccdot%20%5E%7BB%7DT_%7BC%7D%20%5Ccdot%20%5E%7BC%7Dp)
    * keep the vantage point stable and move p using translation for each move (remeber right to left mult)
    > ![i](https://latex.codecogs.com/gif.latex?T_%7B1%7D%20%5Ccdot%20T_%7B2%7D%20%5Ccdot%20p_%7Bold%7D%20%3D%20p_%7Bnew%7D)
* Mathematicaly they are the same
* The rotation matrix for 3D 
    * is always 3by3
    * has determinat = 1
    * is orthonormal
    * *its inverse is its transpose
* the translation matrix for 3D
    * is always 3x1
* the bottom of transform matrix filed with
    * zeros in 1by3 spots
    * bottom right corner 1 in1by1

### 2.8 Transform Inverse, Rotation Representations Part I

* We start from the general representation of trnansform
* The the general fom of transform inverse is
> ![t](https://latex.codecogs.com/gif.latex?T%5E%7B-1%7D%3D%5Cbegin%7Bbmatrix%7D%20R%5ET%20%26%20-R%5E%7BT%7Dt%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D)
* also the folloing holds for the trasform (we have seen that)
> ![t](https://latex.codecogs.com/gif.latex?T%5E%7B-1%7D%5Ccdot%20T%20%3D%20T%20%5Ccdot%20T%5E%7B-1%7D%20%3D%20i)
* the identity matrix i is a valid transform matrix
* also the transpose of the rotation matrix is its transpose (we know that)
* we have seen in a previous lecture the 3D rotation matrices along the 3 axes
* we need to be able to represent a rotation on an arbitrary axis. not necessarily x, y or z
* if we want the 3D rotation matrix across an arbitrary axes a the rotation matrix will follow all the rules we know so far
    * a 3x3 matrix
    * orthonormal
* if we have axis and the angle we are rotating by we can represent rotation in other ways
* if we have the ax,ay,az of axes a and say the angle is α we have what we need to define the rotation
* we might be given elementary rotations. how much we rotate around x,y and z (rx,ry,rz) and combine these elementary rotations into a single big rotation. 
* the elementary rotations are refered as EULER angles (λx,λy,λz)
* Regardless of how we specify a rotation, any rotation in space is equivalent to rotating around a single axis
* using the 3 elementary rotations (EULER) is equivalent to  a single rotation on a single arbitrary axes by an angle
* in some domains (eg aviation roll,pitch,yaw) using the elementary rotations (EULER angles) makes more sense
    * if the airplane is along the x axis
    * roll is λx
    * pitch is λy
    * yaw is λz
* elementary rotations do not follow the x,y,z pattern. thaey can be csahined in any orderlike λx,λy,λx
* The last rotation representation is as a UNIT QUATERNION (gx,gy,gz,gw).
*   * it has 4 elements (gx,gy,gz,gw)
*   * its normal is 1
* So in total we have 4 different representations of rotation which are interchangeable
    * from a Rotation matrx we can compute the single axis rotation and the angle or the roll,pitch,yaw or a quaternion
    * all computer libraries have methods to allow us to move between these forms

### 2.9 Rotation Representations Part II

* Advantages and Disadvandages of each Rotation Representation Format
* Rotation matrix:
    * (+) easy intuition
    * (+) easy to chain rotations
    * (-) memory consumption (9 nums)
    * (-) not CPU friendly
* Axis Angle:
    * (+) easy intuition
    * (-) difficult to chain rotations
    * ( ) memory consumption (4 nums)
    * ( ) not very  CPU friendly
* Elementary Rotation (Euler Angles):
    * (+) easy intuition
    * ( ) not so easy to chain rotations
    * (+) memory consumption (3 nums)
    * ( ) not very CPU friendly
* Unit Quaternion:
    * (-) difficult intuition
    * (+) easy to chain rotations
    * ( ) memory consumption (4 nums)
    * (+) very CPU cycle friendly
* So the verdict is: Rotation matrices are best for Human understanding. Quaternions are best for computers.
* How many numbers we need to uniquelly define our rotation in space?
* How many intrinsic degres of freedom are there?
    * Rot.matrix 9 nums are not independent as the table is orthonormal
    * Axes-angle 4 nums are not independent because the axis is normalized,  its just direction in space, the magnitude of the vector is meaningless. so the axis has 2 only 2 independent variables + the angle. 3 independent variables
    * Elementary rotations 3 EULER angles are independent
    * Unit quaternion has 3 independent variables and 1 dependent derived by the fact thats a unit quaternion
* So the answer to the question is 3. Any rotation in space has 3 independent degrees of freedom
* For the complete Transfor matrix in 3D space. we know that
    * Rotation matrix has 3 degrees of freedom
    * Translation vector has 3 degrees of freedom
* So a complete Transform in 3D space has 6 D.O.F.

### 2.10 Transforms in ROS, the TF Library

* we will talk about TF the ROS library for transformations an more specificly its 2nd version TF2
* TF manages a tree of transforms
* we have talked about coordinate frames that are relative to each other with transforms to go from one to the other
* TF2 keeps a tree so that it knows e.g that A is our base transform and B is defined relative to B so its places as a branch to A and C and D are defined relative to B so are its children
* As we have seen transformations have a direction between frames so they are represented as arrows in the tree
* As we define the transforms TF builds the tree for us
* at any point we can ask TF say whats the transform from D to A and it will calculate it for us
* we visit [TF ROS wiki](http://wiki.ros.org/tf2) and read the tutorial
* we will run some Python ROS code that uses TF
```
#!/usr/bin/env python
import rospy
import numpy
import tf
import tf2_ros
import geometry_msgs.msg

def message_from_transform(T):
	msg = geometry_msgs.msg.Transform()
	q = tf.transformations.quaternion_from_matrix(T)
	translation = tf.transformations.translation_from_matrix(T)
	msg.translation.x = translation[0]
	msg.translation.y = translation[1]
	msg.translation.z = translation[2]
	msg.rotation.x = q[0]
	msg.rotation.y = q[1]
	msg.rotation.z = q[2]
	msg.rotation.w = q[3]
	return msg

def publish_transforms():
	T1 = tf.transformations.concatenate_matrices(
		tf.transformations.translation_matrix((1.0,1.0,0.0)),
		tf.transformations.quaternion_matrix(
			tf.transformations.quaternion_from_euler(1.0,1.0,1.0)		
		)		
	)
	T1_stamped = geometry_msgs.msg.TransformStamped()
	T1_stamped.header.stamp = rospy.Time.now()
	T1_stamped.header.frame_id = "world"
	T1_stamped.child_frame_id = "F1"
	T1_stamped.transform = message_from_transform(T1)
	br.sendTransform(T1_stamped)

	T2 = tf.transformations.concatenate_matrices(
		tf.transformations.translation_matrix((1.0,0.0,0.0)),
		tf.transformations.quaternion_matrix(
			tf.transformations.quaternion_about_axis(1.57,(1,0,0))		
		)		
	)
	T2_stamped = geometry_msgs.msg.TransformStamped()
	T2_stamped.header.stamp = rospy.Time.now()
	T2_stamped.header.frame_id = "F1"
	T2_stamped.child_frame_id = "F2"
	T2_stamped.transform = message_from_transform(T2)
	br.sendTransform(T2_stamped)

# T2_inverse = tf.transformations.inverse_matrix(T2)
# T3_stamped = geometry_msgs.msg.TransformStamped()
# T3_stamped.header.stamp = rospy.Time.now()
# T3_stamped.header.frame_id = "F2"
# T3_stamped.child_frame_id = "F3"
# T3_stamped.transform = message_from_transform(T2_inverse)
# br.sendTransform(T3_stamped)
	
# T1_inverse = tf.transformations.inverse_matrix(T1)
# T4_stamped = geometry_msgs.msg.TransformStamped()
# T4_stamped.header.stamp = rospy.Time.now()
# T4_stamped.header.frame_id = "F3"
# T4_stamped.child_frame_id = "F4"
# T4_stamped.transform = message_from_transform(T1_inverse)
# br.sendTransform(T4_stamped)

if __name__ == "__main__":
	rospy.init_node("tf2_examples")
	
	br = tf2_ros.TransformBroadcaster()
	rospy.sleep(0.5)

	while not rospy.is_shutdown():
		publish_transforms()
		rospy.sleep(0.5)
```
* in the code above:
	* we import tf and geometry msgs type
	* we get a transformation matrix and generate various rotation etc using tf
	* we use the tf2_ros package and broadcat transformations to the rest of the ROS system
	* again we use tf transformations related methods to manipulate matrices
	* the publish message method builds a transformation matrix using tf
	* note how we get quaternions from other formats ofr perfomrmance
	* the transform is stamped and the relationship established before we build the message and broadcast it
	* also note how tf builds the transformation matrix from a translation vector and a rotation matrix
* to run the code:
	* start roscore `roscore`
	* inside the catkin workspace src '/catkin_ws/src/' run `catkin_create_pkg tf2_examples roscpp rospy tf tf2_ros geometry_msgs` to create the package adding the libs we will use
	* we build the package. in /catkin_ws we run `catkin_make`
	* in catkin_ws/src/tf2_examples we `mkdir scripts` to put our python script
	* we cd into scripts and create anew python script and make it executable
```
touch tf2_examples.py
chmod +x tf2_examples.py
```
	* cp the code into it
* run it with `rosrun tf_examples tf2_examples.py`
* with `rostopic list` we see the active topics. our node is boradcasting in \tf
* we listen to it with `rostopic echo \tf` while running the script and see the transformations being broadcasted. success!!!
* to visualize what we do we start visualizer with `rosrun rviz rviz` before running our script. also in rviz we need to add TF. also name fixed frame 'world'
* before running rviz run `rosrun tf static_transform_publisher 0 0 0 0 0 0 1 world map 100` to set the world frame
* vizualizer shows our transforms in 3d space the the coordinate frames are shown with their axes in RGB color. Red =x Green=y Blue=z
* also we see the world coordinate frame in the 3d space and confirm that our transforms start from it
* we see that our applied transformations have logical explanation in 3d space visualizer
* we see that with tf library its very easy to invert transformations and go back so F3 ends up on F1 and F4 on world frame
* to run the inverse transformation we uncomment the inverse code in a new py file 'tf2_examples_inverse.py' and run it `rosrun tf2_examples tf2_examples_inverse`
* what our initial code 'tf2_examples' does is:
> ![t](https://latex.codecogs.com/gif.latex?%5E%7Bworld%7DT_%7BF1%7D%5Ccdot%20%5E%7BF1%7DT_%7BF2%7D)
* what our second code file 'tf2_examples_inverse.py' does is:
> ![t](https://latex.codecogs.com/gif.latex?%5E%7Bworld%7DT_%7BF1%7D%5Ccdot%20%5E%7BF1%7DT_%7BF2%7D%5Ccdot%20%5E%7BF2%7DT_%7BF3%7D%5Ccdot%20%5E%7BF3%7DT_%7BF4%7D%20%3D%20T_%7B1%7D%5Ccdot%20T_%7B1%7D%5E%7B-1%7D%5Ccdot%20T_%7B2%7D%5Ccdot%20T_%7B2%7D%5E%7B-1%7D%20%3D%20i)
* note that transformations.py lib is availalble in [github](https://github.com/ros/geometry/blob/melodic-devel/tf/src/tf/transformations.py)
* we can review the avialble methods

### Project 2

**Description**
* This project will introduce you to 'tf', the ROS framework for handling transforms. Please make sure you have read the entry on this package on the ROS wiki. In this project we consider a ROS ecosystem, which consists of a robot with a camera mounted on it as well as an object. To describe the poses of all these items, we define the following coordinate frames:
    * A base coordinate frame called 'base'
    * A robot coordinate frame  called 'robot'
    * A camera coordinate frame called 'camera'
    * An object coordinate frame 'object'
* The following relationships are true:
* 1. The transform from the 'base' coordinate frame to the 'object' coordinate frame consists of a rotation expressed as (roll, pitch, yaw) of (0.79, 0.0, 0.79) followed by a translation of 1.0m along the resulting y-axis and 1.0m along the resulting z-axis. 
* 2. The transform from the 'base' coordinate frame to the 'robot' coordinate frame consists of a rotation around the z-axis by 1.5 radians followed by a translation along the resulting y-axis of -1.0m. 
* 3. The transform from the 'robot' coordinate frame to the 'camera' coordinate frame must be defined as follows:
    * The translation component of this transform is (0.0, 0.1, 0.1)
    * The rotation component of this transform must be set such that the camera is pointing directly at the object. In other words, the x-axis of the 'camera' coordinate frame must be pointing directly at the origin of the 'object' coordinate frame. 
* In the provided solution.py write a ROS node that publishes the following transforms to TF:
    * The transform from the 'base' coordinate frame to the 'object' coordinate frame 
    * The transform from the 'base' coordinate frame to the 'robot' coordinate frame 
    * The transform from the 'robot' coordinate frame to the 'camera' coordinate frame
**Additional Information**
* You will probably want to make use of the transformations.py library. The documentation for using that is in the library itself; you can reference the version used with ROS online on Github (be careful - other versions of this file exist on the Internet, so if you just Google for it you might get the wrong one).
* For a rotation expressed as roll-pitch-yaw, you can use the quaternion_from_euler() or euler_matrix() functions with the default axes convention - i.e. quaternion_from_euler(roll_value, pitch_value, yaw_value). You can also use the code in tf_examples.py for guidance.
* Be careful about the order of operations. If a transform specifies that the rotation must happen first, followed by the translation (e.g. at points 1. and 2. above), make sure to follow that.
* The transforms must be published in a continuous loop at a rate of 10Hz or more. The skeleton code you are provided already does that, so all you need to do is edit the publish_transforms() function to fill in the transforms with the appropriate values. 
* This assignment also includes some visual feedback. Once you have sourced setup_project2.sh you can click the 'Connect' button. You will see an interactive visualization containing a cube, a cylinder and an arrow. Initially they are all placed at the origin (and the cube will occlude the cylinder).
* Once you run your code, these bodies will position themselves in space according to the transforms your code is publishing. The cylinder denotes the object, the cube and arrow the robot and camera respectively. If your code works correctly, you should see the arrow point out of the cube directly at the cylinder. Here is an example of the correct output (note that the colored axes show you the location of the base coordinate frame with the usual convention: x-red, y-green, z-blue):

![image](https://drive.google.com/uc?export=view&id=1TbYxJsFdQHTc2AT6gxDlQXtbI8ZQoQu5)

**Setup**
* As always, make sure to `source setup_project2.sh`  before trying to invoke any ROS commands (catkin_make, roscd, etc.). This will also start a roscore for your session. Please do not start your own roscore.
* As mentioned above, after you have sourced `setup_project2.sh` simply run your `node rosrun project2_solution solution.py`. After that, you can click the 'Canvas' button on the right corner and then click the 'Connect' button. You will see an interactive visualization of the transforms in the assignment (if you're curious, this was created using [ROS Markers](http://wiki.ros.org/rviz/DisplayTypes/Marker)).
* Each of the three transforms that you need to publish is worth 5 points. For each transform, you will get the points only if the transform you publish is correct in its entirety (within numerical precision) - no partial credit if only the rotation part is correct, or only the translation, etc. 
**How do run this project in my own Ubuntu machine?**
* Launch Project 2, then in Vocareum click Actions>Download Starter code. This will download all the files you need to make the project run locally in your computer.
* IGNORE all the files outside the catkin_ws folder. You do not need these in your local machine 
* The downloaded files are structured as a catkin workspace. You can either use this structure directly (as downloaded) and build the workspace using the "catkin_make" command or use whatever catkin workspace you already had, and just copy the packages inside your own src folder. If you are having troubles with this, you should review the first ROS tutorial "Installing and configuring your ROS Environment".
* Once you have a catkin workspace with the packages inside the src folder, you are ready to work on your project without having to make any changes in any of the files. 
* NOTE: You can source both your ROS distribution and your catking workspace automatically everytime you open up a terminal automatically by editing the ~/.bashrc file in your home directory. For example if your ROS distribution is Indigo, and your catkin workspace is called "robotics_ws" (and is located in your home directory) then you can add the following at the end of your .bashrc file:
```
source /opt/ros/kinetic/setup.bash
echo "ROS Kinetic was sourced"
source ~/robotics_ws/devel/setup.bash
echo "robotics_ws workspace was sourced"
```
* This way every time you open up a terminal, you will already have your workspace sourced, such that ROS will have knowledge of the packages there.
* To run the project, open up a terminal and fire up a roscore (just type "roscore"). Before moving forward, if you haven't followed the instructions on step 5, you will need to source ROS and the catking workspace every time you open a new terminal. On another 2 separate terminals you need to run the scripts in each package: "rosrun marker_publisher marker_publisher" and "rosrun project2_solution solution.py". Now, to visualize the markers we need to launch rviz. In a new terminal type "rosrun rviz rviz". First thing you need to do is change the Fixed Frame option on the left of the UI. Select "base_frame", and notice that the Global Status now reads "Ok". Now we need to add the information we want to be displayed. Click Add and on the popup screen select the tab "By topic". Here you will see the topic /visualization_marker>Marker. Select it and then you should be able to see the block, cylinder and arrow. You can also add the item "TF" if you want to see a visual representation of the frames.
**How to aim the camera?**
* Hint: There is a simple geometrical argument that can help you rotate the x-axis of the arrow to point at the cylinder. Calculate the vector pointing from the camera to the object, use the dot and cross products to deduce the angle and axis to rotate around.

## Week 3: Robot Arms - Forward

### 3.1 Robot Arms Introduction

* Robot Arms revolutionized manufacturing (cars,consumer electronics)
* not so visible to public eye
* [Kuka](https://www.kuka.com/) robots have
    * multiple custom end effectors to perform varius tasks 
    * repetitive tasks. delivers end effector to same position again and again
    * switches grippers
    * forward trajectory and reverse again and again
* Robot arms exceed humans
    * precision
    * tireless
    * speed (turnaround time)
    * strength
    * smooth operation
    * cooperations/synchronization
* [FANUC](https://www.fanuc.eu/uk/en)
    * huge payloads
* Robot Arms can work as 3D Printers
* Our first task is to understand how the Robot arms executes the instructions of moving in 3d space

### 3.2 Kinematic Chains and Forward Kinematics

* What is a Kinematic Chain (aka Robot Arm in Kimematic Analysis):
    *  Asequence of Links and Joints (links connected by joints)
    *  Links are the rigid components that comprise the arm
    *  The Joints are articulations, things that can move
* when we do kinematic analysis joints are modeled as having a single degree of freedom (DOF). 1 direction of movement
* if a real robot joint has >1 DOF we model it as a sequence of joints i Kinematic Analysis
* in kinematic analysis we have the folloing 2 types to model joints
    * Revolute Joint Type: The joit axis is the axis around which we rotate and the joint value (current position) is the rotation angle
    * Prismatic Joint Type: like a hydravlic cylinder. the joint axis is the axis along which we translate. the joint value (current position) is the translation distance
* q is used to represent the joint value and 
    * we use q(d) for prismatic joints
    * we use q(Θ) for revolute joints
* we can have kinematic chains connected to other kinematic chains
* In many applications we just care about a robot arm ability to deliver its end effector at a certain location in space
* What we do in that case is set some values to joints so that the end effector get to the desired location. this analysis is called forward kinematics
* So the Forward Kinematic Analysis asks given the values of all the joints, where my end effector will end up in space. also it considers about the surrounding space not allowing any movement so that links could hit an obstacle
* we attach a coordinate frame to every link assuming that every link has a coordinate frame attached to it. also the end effector has a coordinate frame attached to its end point
* So we rephrase the question of Forward Kinematics. given certain joint values where in space do the coordinate frames attached to the links end up in space?
* Whats the transform of the base coordinate frame to the end effector coordinate frame and all other links defined coordinate frames
* our convention is that going from base to end effector Joint i (Ji) connects Link i-1 (Li-1) to Link i (Li). Li has a coordinate fram i attached to its end {i}
    * for n Joints
    * we have n+1 Links
    * and {n+1} coordinate frames
    * Ji moves Li
    * coord frame {i} is at the tip of Li
    * coord frame {n} is the end effector
* If we now how to compute where the end effoctor coord frame ends up we can compute where all intermendiate coordinate frames end up
* in essensce the question of Forward Kinematics is what is the transofrm from the base to the end effector coordinate frame
> ![i](https://latex.codecogs.com/gif.latex?%5E%7Bb%7DT_%7Bee%7D%3D%3F)
* To compute it we chain robot arm transforms of all robot arm koints and links in sequence from base to end effctor. not that the Joint transforms are not  fixed but demend on the joint value q. otherqise the robot arm whould be rigid
> ![t](https://latex.codecogs.com/gif.latex?%5E%7Bb%7DT_%7Bee%7D%3DT_%7BL0%7D%5Ccdot%20T_%7BJ1%7D%28q_%7B1%7D%29%5Ccdot%20T_%7BL1%7D%5Ccdot%20T_%7BJ2%7D%28q_%7B2%7D%29%5Ccdot%20T_%7BL2%7D%5Ccdot%20T_%7BJ3%7D%28q_%7B3%7D%29%5Ccdot%20T_%7BL3%7D%20%3D%20T_%7BL0%7D%5Ccdot%5Cprod_%7Bi%3D1%7D%5E%7Bn%7DT_%7BJi%7D%28q_%7Bi%7D%29T_%7BLi%7D)
* some robots miss the TLo as tey dont have rigid base
* So the FW Kinematics Equation has:
    * Fixed Transforms for the rigid parts (Links)
    * Variable Transforms for the moving parts (Joints) that change at run-time
* When a Robot moves around it will always tell us what its current joint values are through sensors. so the Forward Kinematics Equation will be calculated at real time
* The Robot mnanufacturer gives the transformations so we can compute the forward kinematics
* for the rest of the system the robot is just something broadcasting its joint values qi and also accepting commands to go to specific joints values

### 3.3 Forward Kinematics: URDF notation

* We write again the Forward Kinematics Full Notation
> ![t](https://latex.codecogs.com/gif.latex?%5E%7Bb%7DT_%7Bee%7D%3D%20T_%7BL0%7D%5Ccdot%5Cprod_%7Bi%3D1%7D%5E%7Bn%7DT_%7BJi%7D%28q_%7Bi%7D%29%5Ccdot%20T_%7BLi%7D)
* The robot manufacturer in the device manual will give us the TLi and how to compute TJi based on the joint values qi
* This notation is not very much used in industry but is used in robot research and ROS
* An example of a notation is URDF(Universal Robot Decription Format)
* It is a format that allows us to define a robot and its general
* we will see the part of [URDF](http://wiki.ros.org/urdf) that can help us describe kinematics
* it uses xml syntax with tags for links and joints
* the kinematic information we care about is wrapped in the joint tags (joint tags have also the joint type)
* the URDF descripition below
    * assumes a base coordinate frame {b}
    * the <origin> tag tells us where the coordinate frame is in relation to the previous coordinate frame. it is a complete transform. it contains the translation part "xyz" and the rotation part in our case in EULER angles (roll,pitch,yaw) "rpy"
    * Tj1 rotates around the z axis according to the angle q1. if q1=0 its on x axis. if its q1>0 it starts pointing to the viewer. if its <0 it points inwards
    * the robot arm is like a human arm locked in the vertical position
```
<robot>
    <link name="L0" />
    <joint name="J1" type="revolute">
        <origin xyz="0 0 0.1" rpy="0 0 0" />
        <axis xyz="0 0 1" />
        <parent link="L0" />
        <child link="L1" />
    </joint>
    <link name="L1" />
    <joint name="J2" type="revolute">
        <origin xyz="0.1 0 0" rpy="0 0 0" />
        <axis xyz="0 0 1" />
        <parent link="L1" />
        <child link="L2" />
    </joint>
    <link name="L2" />
</robot>
```
* the first job when we design a robot in ROS is to make its representation in URDF format
* URdF can contain much more info apart from kinematics such as shape info, inertia, mass, collission info, vision info, sensor info. all these with their respective tags

### 3.4 Forward Kinematics: DH Examples

* DH stands for [Denavit-Hartenberg](https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters) which is a notation used more in the industrial world
* Industry does not use URDF because its verbose and general. when robots came into tthe scene decades ago processors were small so they could not afford it. also URDF is not optimised for computing FW kinematics analytical. to come up with a formula that we can derive by hand that will tell us the transform from base to the end effector
* URDF is good for computers but not for human intutition and manual computations
* DH Notation is old,proven and widely used in industry, its compact
    * it uses conventions, its not general
* The conventions are:
    * the Joint Axis is always the local z axis (axis for Ji is zi-1) so its the z axis for coordinate frame {i-1}
    * Li can be only 1 of 2 things: a translation along the local x axis OR a rotation around the local x-axisthe local x
* With DH notation to calculate 
> ![t](https://latex.codecogs.com/gif.latex?T_%7BJi%7D%28qi%29%5Ccdot%20T_%7BLi%7D)
we need only 4 numbers
    * θi the rotation around local z
    * di the translation across local z
    * ai the translation accross local x
    * αi the rotation around local x
* The 2 transforms chained together always in DH notation look like this
> ![dh](https://latex.codecogs.com/gif.latex?T_%7BJi%7D%28qi%29%5Ccdot%20T_%7BLi%7D%20%3D%20T_%7BROT%7D%28%5Ctheta_%7Bi%7D%2Cz%29%5Ccdot%20T_%7BTRANS%7D%28d_%7Bi%7D%2Cz%29%5Ccdot%20T_%7BTRANS%7D%28a_%7Bi%7D%2Cx%29%5Ccdot%20T_%7BROT%7D%28%5Calpha_%7Bi%7D%2Cx%29)
* In DH notation the joint value q can be either θ οr d param

### 3.5 DH Notation Example: 2-link Planar Robot

* Our first robot is a 2-jointed robot.
* for every joint i we will have θi di ai αi: 
    * for joint 1: θ1=q1, d1=0 a1=0.5m α1=0
    * for joint 2: θ2=q2, d1=0 a1=0.3m α1=0
* How this robot looks like?  
* for simplicity we draw 3D space as 2D assuming z axis of base frame to point towards the viewer
* joint1 is a revolute joint around base z axis. the angle is q1
* the link between joint 1 and 2 is 0.5m along the x axis of joint 1 and it
* again joint 2 is revolute it rotates around z axis by q2 and link 2 is 0.3 on its x axis. 
* this is where end effector is 
* this is a planar 2-link robot. a 2d robot
* We see that DH notation is very intuitive for humans. conventions help on this
* depending on the joint type. the joint value qi goes to θ or d respectively, the other is fixed
* a and α are fixed as they represent the link
* Manufacturer (KUKA, FANUC) gives the DH table and from this we derive the robot kinematics
* The true challenge is to formally compute the Transform from Base to EndEffector using the DH table
* We start with the FW Kinematic analysis using DH notation for each Joint
<p align="center"><img src="/tex/94c1bc37c3623f241cf320840074d6ea.svg?invert_in_darkmode&sanitize=true" align=middle width=409.7404509pt height=16.438356pt/></p>

* what this means for the 2-axis planar robot with the DH params we have seen ? we chain the transforms
<p align="center"><img src="/tex/1605da9de51f61e2dacc0370e5949034.svg?invert_in_darkmode&sanitize=true" align=middle width=175.86145994999998pt height=16.438356pt/></p>
<p align="center"><img src="/tex/bf481c3eb009ba29f02f65dd490a8d91.svg?invert_in_darkmode&sanitize=true" align=middle width=1359.6071642999998pt height=59.1786591pt/></p>

* to get to c12 s12 we use trigonometric rules

<p align="center"><img src="/tex/56034d267cdfbb4a92babbec5565bf5c.svg?invert_in_darkmode&sanitize=true" align=middle width=228.35023694999998pt height=16.438356pt/></p>

* we take a look on our full derived translation matrix and see if it makes sense. if it follows the rules
    * we have a rot matrix with a rotation q1+q2 around z and that makes sense
    * the translation part makes sense trigonometrically according to our sketch

###  3.6 DH Notation Example: SCARA Robot

* we will see another robot fully defined in DH notation
* the specification of the robot is:
    * i=0: θ0=0 d0=0.5 a0=0 α0=0
    * i=1: θ1=q1 d1=0 a1=0.7 α1=0
    * i=2: θ2=q2 d2=0 a2=0.7 α2=0
    * i=3: θ3=0 d3=q3 a3=0 α3=0
    * i=4: θ4=q4 d4=0 a4=0 α4=0
* we see that it has 5 joints. (acually 4 and one fixed to represent L0 link) the j=0 is a fixed link
* we scetch it starting from base coordinate frame (y points toward viewer)
    * first fixed joint J0 is 0.5m towards the +z axis (L0 link). no rotation
    * second joint J1 rotates around z axis for q1 degrex (new x and new y axis) and L1 link is 0.7m along the rotated J1 x axis with no further fixed rotation
    * third joint J2 rotates around z axis of L1 tip for q2 deg  (z is still not rotated in system) (new x and new y axis) L2 link is 0.7m along the rotated J2 x axis with no further fixed rotation
    * forth joint J3 is a prismation one (varialble translation on rotation) wher q1 is the joint value (distance) along the positive z axis. prismatic joint has no a and α as it serves as a movable link. no rotation of any axis
    * last joint J4 is a rotation around the z axis for q deg no link. so J3 and J4 work together as a movable link with a rotated tip that rotates the end effector
* this robot is an efficient pick and place robot
* we now know how to calculate analytically its transform matrix base->endeffector from DH params
* we will use the ci si notation for cos(θi) sin(θi) and cij sij for cos(θi+θj) and sin(θi+θj)
<p align="center"><img src="/tex/44e7acc47000226920120e80d80dbcae.svg?invert_in_darkmode&sanitize=true" align=middle width=926.03717745pt height=78.9048876pt/></p>

* when we multiply a trasform matrix containing only translation (R=i) with a transform matrix containing only rotation we can just concatenate the matrices. This works ONLY IN THIS ORDER: Pure transaltion followed by pure rotation
* when we have consecutive Pure Translations we can just add them up
<p align="center"><img src="/tex/6e1585d4fbac05a525708e2e32642cc5.svg?invert_in_darkmode&sanitize=true" align=middle width=499.45122480000003pt height=78.9048876pt/></p>

* we can mutliply the matrixes to produce the final transform

### 3.7 Kinematic examples: 6DOF and 7DOF robots

* we will now look ata a more complicated robot with 6 joints aka 6DOF
* all joints are revolute (no prismatic)
* this robot uses a lot α (fixed rotation around x axis)
* such a robot is very common in industrial robotics
* this exact DH description is from a [Staubli]() robot
    * joint1: θ1=q1 d1=0 a1=0 α1=90deg
    * joint2: θ2=q2 d2=0.16 a2=-0.4 α2=0
    * joint3: θ3=q3 d3=-0.14 a3=0 α3=90deg
    * joint4: θ4=q4 d4=0.45 a4=0 α4=90deg
    * joint5: θ5=q5 d5=0 a5=0 α5=-90deg
    * joint6: θ6=q6 d6=0.07 a6=0 α6=0
* we start our attempt to draw the robot arm by draing the base coordinate frame in 3d perspective and z pointing up, x towards vier y points right
* first joint is rotation around z axis by variable angle. the α rotates the frame of joint 90o around x so now z points left and x unchanged and y down
* second joint is rotating arouns the new z by variable angle and is translated on z axis. the next link is translated on x axis negative so points inside the screen
* third joint rotates around the left pointing z by variable angle and translates negative on z. the link is rotated 90deg around x. so the new frame has a z pointing down
* forth joint rotates around the new down pointing z by variable angle and is translated on z axis producing the link. the new frame is fixed rotated around x 90deg so z now points right
* fifth joint rotates around the new z and has no translation on any axes so 4 and 5 is a double joint. it has a fixed rotation on x of -90deg so the new frame has a z pointing down again
* 6th joint rotates around the new z axis and translates on it by 0.07
* [Graspit](https://graspit-simulator.github.io/) is an open source simulator for robotic hands and arms. if we model this arm we see the degrees of freedon and can move it
* a is used in these robots heavily because we need the next z axis to point to the right dir
* we will look at a similar robot in ROS
* he uses a robotsim and then rviz to visualize and enables a robotmodel and then uses a python script to command it
* rviz actually uses TF info underneath. we can enable it and see the joint frames in realtime
* so as we have seen TF is used broadcasting the frames or the rviz to consume
* there is a ROS module computing forward kinematics based on the input from the applet. the transformations are published to TF . rviz listens to TF and visuzalizes

### 3.8 Recap

* Kinematic chains are collections of links and joits
* diffrent values of the joints (joint values or variables) mans the robot is moving
* the transformation params or robot model params are given in URDFor DH format

### Project 3

**Description**
* In this project you will implement the forward kinematics for a robot arm defined in a URDF file and running in a ROS environment.
* The setup contains a "simulated" robot that continuously publishes its own joint values. After you have run through the Setup instructions (see below), you can check that the robot is indeed publishing its joint values by using the 'rostopic echo /joint_states' command. However, that is not enough for the robot to be correctly displayed: a forward kinematics module must use the joint values to compute the transforms from the world coordinate frame to each link of the robot. This is the code you must fill in.
* Your job will be to complete the code 'solution.py' in the 'forward_kinematics' package provided to you. When you familiarize yourself with the starter code you will see that the 'ForwardsKinematics' class subscribes to the topic 'joint_states' and publishes transforms to 'tf'. It also loads a URDF description of the robot from the ROS parameter server. You will only have to edit 'solution.py' and fill in the compute_transforms function. If you want, you can also peruse the rest of the skeleton we provide to get an even better understanding of what is going on behind the scenes.
* Every time the subscribed receives new joint values, we do some prep work for you. We unpack from the URDF all the data you will need, including the structure of the robot arm as lists of joint objects and link names. Then, we pass this data, along with the joint values, to the compute_transforms function which you must fill in.

**The 'compute_transforms' function**
* This is the function that performs the main forward kinematics computation. It accepts as parameters all the information needed about the joints and links of the robot, as well as the current values of all the joints, and must compute and return the transforms from the world frame to all the links, ready to be published through tf.
* Parameters are as follows:
* link_names: a list with all the names of the robot's links, ordered from proximal to distal. These are also the names of the link's respective coordinate frame. In other words, the transform from the world to link i should be published with world_link as the parent frame and link_names[i] as the child frame.    
* joints: a list of all the joints of the robot, in the same order as the links listed above. Each entry in this list is an object which contains the following fields:
    * joint.origin.xyz: the translation from the frame of the previous joint to this one
    * joint.origin.rpy: the rotation from the frame of the previous joint to this one, in ROLL-PITCH-YAW XYZ convention
    * joint.type: either 'fixed' or 'revolute'. A fixed joint does not move; it is meant to contain a static transform. 
    * joint.name: the name of the current joint in the robot description
    * joint.axis: (only if type is 'revolute') the axis of rotation of the joint
    * joint_values contains information about the current joint values in the robot. It contains information about all the joints, and the ordering can vary, so we must find the relevant value  for a particular joint you are considering. We can use the following fields:
* joint_values.name: a list of the names of all the joints in the robot;
    * joint_values.position: a list of the current values of all the joints in the robot, in the same order as the names in the list above. To find the value of the joint we care about, we must find its name in the name list, then take the value found at the same index in the position list.
* The function must return one tf message. The transforms field of this message must list all the transforms from the world coordinate frame to the links of the robot. In other words, when you are done, all_transforms.transforms must contain a list in which you must place all the transforms from the world_link coordinate frame to each of the coordinate frames listed in link_names. You can use the convert_to_message function (defined above) for a convenient way to create a tf message from a transformation matrix.

**Setup**
* Similarly to the first two projects, please make sure you source the 'setup_project3.sh' before  you attempt to run your code. This starts a roscore and loads the robots URDF into the ROS parameter server. After you have done that, you can press the 'Connect' button and you should see the robot arm with all its links placed at the origin. This is because no transform tree is being published and ROS does not know where to place the links.
![image](http://roam.me.columbia.edu/files/seasroamlab/imagecache/103x_P3_1.png)

* The setup script will also start a nodes that you can find in the 'robot_mover' and 'robot_sim' package. These node publish joint values on the 'joint_states' topic, which your forward kinematics code subscribes to. All that is left for you to do is to run your completed code. If you have done everything correctly, you should see the robot arm move back and forth in a physically correct fashion. 
![ιμαγε](http://roam.me.columbia.edu/files/seasroamlab/imagecache/103x_P3_2.png)

* When you run solution.py, you will get a Warning along the lines of "Unknown tag: comScalar element defined multiple times...". You can safely ignore this.
* If you get a notification that the websocket connection has closed that means that the connection between ROS and the Canvas has broken down. You will have to reload the page and source the setup script again before ROS can use the Canvas again.

**Resources and Hints**
* It will help to get familiar with  the [URDF documentation](http://wiki.ros.org/urdf). In particular, the documentation for the [URDF Joint](http://wiki.ros.org/urdf/XML/joint) element will be very helpful in understanding the nature of the joint object that is being passed to the compute_transforms function, and what you must do with the data in each joint object.
* Remember that you must compute (and publish) the transform from the world coordinate frame (called world_link) to each link of the robot. However, the URDF tells you the transform from one link to the next one in the chain (through the joint between them). Thus, one way to complete the assignment is in iterative fashion: assuming you have compute the transform from the world_link coordinate frame to link i, you just need to update that with the transform from link i to link i+1 and you now have the transform from the world_link frame to link i+1.

### Project 3 FAQ

* How do run this project in my own Ubuntu machine?
    * Launch Project 3, then in Vocareum click Actions>Download Starter code. This will download all the files you need to make the project run locally in your computer.
    * Install the needed ROS package(s). Run the following lines on your terminal:
```
sudo apt-get update
sudo apt-get install ros-kinetic-urdfdom-py
```

* Replace kinetic with the ROS version that you are running on your local machine.
    * IGNORE all the files other than 'catkin_ws' and 'kuka_lwr_arm.urdf'. Copy the folder catkin_ws to your home directory (you can rename it project3 if you want). Also put the file 'kuka_lwr_arm.urdf' in the home directory.
    * The downloaded files are structured as a catkin workspace. Navigate to the folder catkin_ws in your home directory using "cd catkin_ws" or whatever name you gave the workspace ("cd project3"). If you are running ROS Kinetic you need to modify the CMakeList.txt file in the robot sim package before running catking_make (see note in the last FAQ bullet point). Once inside your catkin workspace, run the command "catkin_make".
    * If you are having troubles with this, you should review the first ROS tutorial "Installing and configuring your ROS Environment".
    * At this point if the catkin_make command was successful, you are ready to work on your project without having to make any changes in any of the files. 
    * NOTE: You can source both your ROS distribution and your catkin workspace automatically everytime you open up a terminal automatically by editing the ~/.bashrc file in your home directory. For example if your ROS distribution is Kinetic, and your catkin workspace is called "project3_ws" (and is located in your home directory) then you can add the following at the end of your .bashrc file:
```
source /opt/ros/kinetic/setup.bash
echo "ROS Kinetic was sourced"
source ~/project3_ws/devel/setup.bash
echo "project3_ws workspace was sourced"
```
    This way every time you open up a terminal, you will already have your workspace sourced, such that ROS will have knowledge of the packages there.
    * Before moving forward, if you haven't followed the instructions on step 6, you will need to source ROS and the catkin workspace every time you open a new terminal. To run the project, first open up a terminal and type "roscore". In the second terminal (remember to source ROS and the catkin workspace if you didn't do step 6)  run "rosparam set robot_description --textfile kuka_lwr_arm.urdf", followed by "rosrun robot_sim robot_sim_bringup".
    * On another 2 separate terminals you need to run the scripts for the robot mover and the your solution in forward kinematics : "rosrun robot_mover mover" and "rosrun forward_kinematics solution.py". Note that you can find these lines from setup_project3.sh in the starter code.
    * Now we can open up Rviz using "rosrun rviz rviz". Inside Rviz, first change the Fixed Frame to "world_link" (you might not be able to do this until you start writing your solution code since there will not be any TF for "world_link"). Then click Add and select RobotModel from the list of options. At this point if you code works, you should see the robot arm rendered and moving in a coherent way back and forth from an upright position to a another predetermined pose. You can also see the transforms if you select Add > TF. 

## Week 4: Robot Arms - Inverse Kinematics

### 4.1 Analytical IK, Planar Robot Example

* Inverse Kinematics is the opposite problem of Formward Kinematics
* Forward Kinematics problem is: Having the robot description and the Joint Values to calculate the Trasform Matrix from Base Frame to EndEffector Frame AKA. the relative position of End Effector
<p align="center"><img src="/tex/7106c5a97bd6b42affbbf9cd9e4ab9d2.svg?invert_in_darkmode&sanitize=true" align=middle width=67.99830675pt height=26.07647955pt/></p>

* In real robot applications we care much more about the inverse problem: we know where the target object is (the desired end effector position). this is the relative position to the base frame which we know so we know the Trasform matrix to go from base to end effector. what we dont know are the joint values
<p align="center"><img src="/tex/841c3c6c1bfddf7db93626577e1391d7.svg?invert_in_darkmode&sanitize=true" align=middle width=67.0644447pt height=26.07647955pt/></p>

* We will see analytical methods to compute the joint values
* The course of action in real life robotics is 
    * Design the Mechanical of Robot 
    * Measure it
    * Derive DH params
    * Calculate FW kinematics
    * Calculate INV kinematics
* we will start simple in 2D using the planar 2 link robot we ve seen in previous chapter
* The DH params of the 2-Link Planar Robot are:
    * Joint1: θ1=q1 d1=0 a1=0.5 α1=0
    * Joint2: θ2=q2 d2=0 a2=0.3 α2=0
* we start from base frame. x->right y->up z-> to viewer
* first joint is q1=Θ1 around z axis so new x axis is rotated by θ1. the link is a translation of 0.5m on the new x axis
* joint is a q2=Θ2 rotation around the z axis. so new x axis is the old rotated by θ1. the link is a translation of 0.3m on the new x axis
* we calculate first transform matrix from nbase to endeffector (Forward Kinematic) which we have doen in previous chapter:
<p align="center"><img src="/tex/b77bcefd5418f4a09613a17b51f12783.svg?invert_in_darkmode&sanitize=true" align=middle width=987.2226341999999pt height=59.1786591pt/></p>

* remember that the following convention is used for trigonometric methods::
<p align="center"><img src="/tex/00afb99363be23587ae1fac241e43170.svg?invert_in_darkmode&sanitize=true" align=middle width=275.5416972pt height=16.438356pt/></p>
<p align="center"><img src="/tex/24d8de1242992f80ba568cc55baff3a7.svg?invert_in_darkmode&sanitize=true" align=middle width=275.49025845pt height=16.438356pt/></p>

* say now we want our end effector at position x=a,y=b we dont care about the orientation of end effector coordinate frame yet just the position. therefore we use only the translation part of the transform matrix frpm base to end effector to derive the joint values (angles θ1 θ2)
<p align="center"><img src="/tex/65d9964d7f04be46227d6fac05b09643.svg?invert_in_darkmode&sanitize=true" align=middle width=266.9802927pt height=13.881256950000001pt/></p>

* a good trick to solve the above equation system is to square them both and then add them up
* what we get is 
<p align="center"><img src="/tex/8a2a76fef8f1b225897d3da2eff975e0.svg?invert_in_darkmode&sanitize=true" align=middle width=480.1584612pt height=18.2666319pt/></p>

* we make use of the theorem 
<p align="center"><img src="/tex/2662b434ff963ff244d2c3c85fc433ba.svg?invert_in_darkmode&sanitize=true" align=middle width=149.08671195pt height=18.312383099999998pt/></p>

* using this and the main trigornometric methods for θ1+θ2 our equation is simplified
<p align="center"><img src="/tex/ad21da414174930ba41c958bc028bc60.svg?invert_in_darkmode&sanitize=true" align=middle width=353.84316164999996pt height=18.312383099999998pt/></p>

* so we have c2.
<p align="center"><img src="/tex/86a073c2704212e1503b3fee6ed7a4e1.svg?invert_in_darkmode&sanitize=true" align=middle width=138.27759659999998pt height=35.77743345pt/></p>

* the immediate thought it to introduce the solution of c2 into the first equation and solve for c_{1}
* not yet. we have to deal with some cases.also we draw a circle of maximum reach for the robot of radius L1+L2
* CASE 1: if the fraction is > 1 it cannot be a cosine so we have no solutions and theoretically the point is in infinite position outside of max reach
* CASE 2: if the fraction is equal to 1 then c2=1 and we have only 1 solution:
<p align="center"><img src="/tex/931fb9cc664bcedb25f37e3008070616.svg?invert_in_darkmode&sanitize=true" align=middle width=305.52425145pt height=39.452455349999994pt/></p>

* the solution if c2=1 is that arm is always fully stretched out on the max reach circle
* when we get c1=a we are tempted to do q1=acos(a) but in the range [0,2π] there are muliple angles with c1=a not one. same for sin. there is not a unique solution. combining both equations gives a unique solution using arctan2. arctan2 is an arctangent that looks at the quadrant that the angle that should be in
<p align="center"><img src="/tex/13c26fcd1aa559dbf2e098a74d7d4ac3.svg?invert_in_darkmode&sanitize=true" align=middle width=214.09940805pt height=39.452455349999994pt/></p>

* so keep in mind. if only cos or sin is given there are multiple solutions. if both are given there is a unique solution for angle
* CASE 3: fraction is between -1 and 1 so -1< c2 <1 then there are 2 possible solutions
<p align="center"><img src="/tex/5130a3990092288538d0545739750ea8.svg?invert_in_darkmode&sanitize=true" align=middle width=218.51575845pt height=49.315569599999996pt/></p>

* if we use these 2 possible solutions in the original equations we can solve for cos and sin of q1
* this is a valid concept as fir every point inside the max reach circle there are 2 ways for the robot to reach it aka 2 posiible solutions
* CASE 4: when fraction is equal to -1. then c2= -1 so q2=π so again a single possible solution for q1 like what we did for c2=1. 
* in the physical world the external link is folded on the nternal so the robot is on an inner circle with radious L1-L2
* CASE 5: fraction < -1 . we are inside the inner circle in a region impossible to reach so 0 solutions
* so for the 2-Link Planar Robot the workspace is donut shaped limited between the external and the internal max reach circle

### 4.2 Robot Workspaces and IK Solutions

* what if apart from translation vector and point postion we put rotation matrix and orientation of the endeffector point in the mix (OMG!!!!!)
* i want to get the end effector in a certain position in space but also i care about the orientation of it.
* then we will have postion (α,β) and an angle (γ) from the x axis
* we start again from the FW full transform matrix to form the equation
<p align="center"><img src="/tex/b9d605f483002bf231abb0f3e52d8e31.svg?invert_in_darkmode&sanitize=true" align=middle width=393.88655789999996pt height=59.1786591pt/></p>

* we cannot solve this linear system. we ask our system to acheive an end effector pose specified by 3 variables (α,β,γ) which is valis for position and orientation in 2d space... but the only means the robot has at its position to acheive that are 2 joint values (angles) (q1,q2)... its needs a 3rd join
* we validated that the robot can achieve position in the donut shaped worskspace. once α and β are set γ also... the robot has 2DOFs... we need a 3rd DOF a 3rd joint to acheive that
* we ll now dig into 3D space.... (OMG^2) to understand the relationship between DOFs and what the end effector can achieve
* In 2D
    * we talk about the x,y plane
    * variable to define the position=(x,y) and orientation=θ of a point in 2D plane: (x,y,θ) 3vars
    * ROBOTS: #DOF #InvKinem Solutions for arbitrary (x,y,θ)
        * '<3 DOF' => no workspace where the number of solutions is >1. 0 solutions (there always be unachievable space where). no space where the robot can achieve ANY combination of x,y,θ
        * '3 DOF' => 0< FINITE for some workspace if the robot is well designed. still there will be some space where the robot cannot achieve ANY combination of x,y,θ (remember the donut shape)
        * '>3 DOF' => a reduntant robot with a part of workspace where we can have infinite solutions
    * so the sweet spot is 3DOF for 2D robot
* In 3D
    * we talk about full cartesian 3D space (x,y,z)
    * to define the position and orientation of a body in space. 6 variables. 3 for position (x,y,z) and 3 for orientation along the 3 axis (roll,pitch,yaw)
    * ROBOTS: #DOF #InvKinem Solutions for arbitrary (x,y,z,r,p,y)
        * '<6 DOF': 0 solutuions. no space where it can achive any combination of x,y,z,r,p,y
        * '6 DOF': >0, FINITE ways to achive ANY combination in some part of 3d space
        * '>6 DOF': redundant robot, INFINITE solutions at some part of space
    * Sweet spot is 6DOF for 3D space robot
* An example of a robot working in 3d space with '<6 DOF' is the SCARA robot we have seen.
* SCARA robot has 4DOF (3 rotations + 1 transaltion). by design its end effector is always pointing down
* A robot with exactly 6DOF is a [Puma](https://en.wikipedia.org/wiki/Programmable_Universal_Machine_for_Assembly) robot (6 rotations). there is some space where it can place the end effector at any combination of position,orientation
* A human arm has 7 degrees of freedom (3 at elbow,1 ant elbow,3 at wrist) why??? redundancy is useful in natural world to overcome obstacles
* in industrial world where environment is controlled jist 6 is enough (also budgetwise)
* for autonomous robots to release in physical world redundancy is ok to oevercome obstacles
* more than 7DOF only in research

### 4.3 IK Recap

* IK given the end effector position and orientation  find the axis values to achieve it given the robot model
* First calculate FK analytically then use it to get Ik as analytical formula
* 6 DOF for 3d 3DOF for 2D

### 4.4 Analytical IK, Spherical Robot

* we go full mode on 3D space doing Forward and inverse Kinematics on a Spherical Robot.
* The DH params of the robot are
    * Joint1: θ1=q1, d1=0, a1=0, α1=-90deg
    * Joint2: θ2=q2, d2=0, a2=0, α2=90deg
    * Joint3: θ3=0, d3=q3, a3=0, α3=0
    * Joint4: θ4=q4, d4=0, a4=0, α4=0
* we start with designing the robot on paper
* our base coordinate frame is: x=viewer y=right z=up 
* first joint is rotating around z and its next coordinate frame is rotated -90deg around x. so new z points right and new frame is at 0,0,0 position (no link)
* second joint is rotating around new z and its next coordinate frame is rotated 90deg around x. so new z points up and new frame is at 0,0,0 position (no link)
* third joint is prispatic so a variable length link of d=q3 on ze axis. new axis frame is translated but not rotated.
* forth joint is rotating around z. nothing more
* we qive the four joint values an non zero val to visualize the robot. starting with the base frame it looks like a vector of variable length in the space between the 3 positive axis x,y,z
    * the angle of the vector (Link) to the +z axis will be q2
    * the angle of the vector (link) projection on the x,y plane with the +x axis will be q1
    * the length of the vector (link) or prismatic joint will be q3
    * the rotation of end effector around the the vector (link) axis  will be q4
* the workspace is a sphere (sperical robot)
* so always when getting DH params scetch the robot
* Then we go for Forward Kinematics Analysis (note we need a transform matrix for α rotations by calculating cos and sin for π)
<p align="center"><img src="/tex/37a2962ae4e1304e4c7fc1d84d1bb197.svg?invert_in_darkmode&sanitize=true" align=middle width=810.4970246999999pt height=78.9048876pt/></p>

* we do the matrix multiplications (rotation part gets very complicated)  so we assume we care only about the translation of the end effector and get to 
<p align="center"><img src="/tex/b2ed7a7d49bcb205013c3d5bb09de7ed.svg?invert_in_darkmode&sanitize=true" align=middle width=218.0355012pt height=59.1786591pt/></p>

* as we said this T matrix is all about the translation of end effector in space. has nothing to do with the rotation of end effector. thats why q4 is missing
* Now we go to Inverse Kinematics taking into consideration only Position and not Orientation. Orientation is too much for doing an analysis on paper. let tools tackle it
* we need to solve the equation system below for q1,q2,q3
<p align="center"><img src="/tex/583b8874ff38ecf4b98f7b2b99433cec.svg?invert_in_darkmode&sanitize=true" align=middle width=89.3835723pt height=59.178683850000006pt/></p>

* we do the trick of squaring them up and adding them
<p align="center"><img src="/tex/5f3f8c0a79883106b13e53ffcde3594a.svg?invert_in_darkmode&sanitize=true" align=middle width=287.2256739pt height=18.312383099999998pt/></p>
<p align="center"><img src="/tex/3a25c8f16c44e2dbce5846c0501ae08d.svg?invert_in_darkmode&sanitize=true" align=middle width=172.2335604pt height=17.399144399999997pt/></p>

* the negative solution is more for clompleteness. its uncommon to have a prismatic joint extending in reverse direction
* usually in practice we have our system limits. a manufactures always gives joint limits
* so we have 2 SOLUTIONS for q3
* we square x and y and add them to go for q1 q2
<p align="center"><img src="/tex/1fd2dc22adee72ec6b1faded451a5b2f.svg?invert_in_darkmode&sanitize=true" align=middle width=167.62925024999998pt height=18.312383099999998pt/></p>

* for s2 we again have 2 solutions
<p align="center"><img src="/tex/c9e1b6a8aa686eb5c4a6311db1247f29.svg?invert_in_darkmode&sanitize=true" align=middle width=158.88411495pt height=49.315569599999996pt/></p>

* but we have a solutuion for c2 sso we can use atan2
<p align="center"><img src="/tex/1ed65cce7f85a539ddf1758cfef177a6.svg?invert_in_darkmode&sanitize=true" align=middle width=225.28059674999997pt height=32.6705313pt/></p>

* as we have 2 solutions for sine we have also 2 SOLUTIONS for q2. also note that we divide by q3. what if q3 is 0. if q3 is 0 x,y,z is 0 so we end up qith a non realistic situation
* now we calculate q1
<p align="center"><img src="/tex/2eb0e6e8a8835571854353b272b2d728.svg?invert_in_darkmode&sanitize=true" align=middle width=268.26472529999995pt height=32.6705313pt/></p>

* so for q1 we have 1 SOLUTION. so 4 solutions for the IK problem if we apply no joint value limits
* if we accept only q3>0 the we end with 2 solutions => q1,q2,q3 equivalent to q1+π,-q2,q3

### Assignment 1

* Consider the robot described by the D-H table below:
    * Joint1: θ1=0, d1=q1, a1=0, α1=0
    * Joint2: θ2=q2, d2=0, a2=0, α2=-90deg
    * Joint3: θ3=0, d3=q3, a3=0, α3=0
* Consider three possible robot sketches, one of which is a correct representation of the robot defined in this problem:
**Sketch 1:**
![sketch1](http://roam.me.columbia.edu/files/seasroamlab/imagecache/103x_h1_r1c.jpg)
**Sketch 2:**
![sketch2](http://roam.me.columbia.edu/files/seasroamlab/imagecache/103x_h1_r1a.jpg)
**Sketch 3:**
![sketch3](http://roam.me.columbia.edu/files/seasroamlab/imagecache/103x_h1_r1b.jpg)
* Compute the translation part of the Forward Kinematics transform  <sup>b</sup>T<sub>ee</sub>  from the base of the robot to the end-effector. In other words, derive the expressions for  x ,  y  and  z  below:
<p align="center"><img src="/tex/ba6c7b88763fd9cfd597521caf543faf.svg?invert_in_darkmode&sanitize=true" align=middle width=183.80800184999998pt height=79.5616338pt/></p>

* We will use the notation from the lectures
<p align="center"><img src="/tex/34e0a277e6ed953f99dc2372e27774f5.svg?invert_in_darkmode&sanitize=true" align=middle width=198.14837294999998pt height=16.438356pt/></p>

* Assume we require the end-effector to be at position  [a,b,c]T , and we do not care about end-effector orientation. Derive the values for the robot joints  q1 ,  q2  and  q3  such that the end-effector achieves the desired position. Be sure to consider all possible solutions.

### 5.1 Differential Kinematics Introduction

* we start again from FW Kinematics on a typical robot arm with 3 links and 3 joints
* FW kinematics is the task of computing the Transform from base to end effector as a functions of the joint angles:
<p align="center"><img src="/tex/c1fed967270fb580c6eddaa62a5997e6.svg?invert_in_darkmode&sanitize=true" align=middle width=320.13042105pt height=23.5253469pt/></p>

* a very common problem when using robots is move the end effector in a specified direction that we know in cartesian space. say from point1 to point2, where we know the change Δx in cartesian space between point 1 and 2. Δx is the difference of 3D point vectors x=[x,y,z]T. if we care also about the orientation apart from position then x=[x,y,z,roll,pitch,yaw] or x=[x,y,z,rx,ry,rz]T. in that case we want to know how much to change the joint values Δq=?
* say we have a welder robot and we need to weld a body so the welding robot has to follow the contour of the surface. we have the path specified in cartesian space. we need to decide how the joints move in order for the robot to follow this path we have specked in cartesian space....
* What the FW Kinematics does for us is that it tells us that the position and orientation of the robot in cartesian space can be expressed as a function of q x=f(q)
* we know that we can easily go from transform matrix to the relative position(and orientation) of end effector in relation to the base frame
<p align="center"><img src="/tex/60e92aef56bbc770d0f754f46e170aed.svg?invert_in_darkmode&sanitize=true" align=middle width=478.58343225pt height=23.5253469pt/></p>
<p align="center"><img src="/tex/1f156eac0de2b7034fcf7e59b68294f8.svg?invert_in_darkmode&sanitize=true" align=middle width=61.84355265pt height=16.438356pt/></p>

* how to compute Δq. we know that x+Δx=f(q+Δq). we can linearize the function f around the point q
<p align="center"><img src="/tex/4141bedb3f02af9b9ed327f9e006983d.svg?invert_in_darkmode&sanitize=true" align=middle width=264.31589909999997pt height=37.0084374pt/></p>
<p align="center"><img src="/tex/59c6b164512cb4d164587ecbb536006d.svg?invert_in_darkmode&sanitize=true" align=middle width=183.13729665pt height=37.0084374pt/></p>

* so our problem is now the parial derivative of the function on the joint values

### 5.2 Manipulator Jacobian

* we need to differentiate the function f but it takes multidimensional input and output
<p align="center"><img src="/tex/a1ba159f50276a1207067ffaea59ef80.svg?invert_in_darkmode&sanitize=true" align=middle width=109.16825534999998pt height=14.937954899999998pt/></p>
<p align="center"><img src="/tex/86c45070c5fe573e621b1d54db4fd87d.svg?invert_in_darkmode&sanitize=true" align=middle width=149.7988074pt height=14.611878599999999pt/></p>
<p align="center"><img src="/tex/28b22c2641cdbbf56cab275dd47aa956.svg?invert_in_darkmode&sanitize=true" align=middle width=194.59729905pt height=14.611878599999999pt/></p>

* the differentiate functions can be expressed as a m by n matrix of partials
<p align="center"><img src="/tex/bef127627b53d2fe8fe7a1044d704ae0.svg?invert_in_darkmode&sanitize=true" align=middle width=248.16740354999996pt height=64.9991991pt/></p>

* this matrix is called the Jacobian of function f (n columns and m rows)
* the jacobian is the differentiation of function f against q but its valid in aparticular location in input space so the jacobian is a function of q. J(q). as the values of q change so does the matrix
<p align="center"><img src="/tex/445d6dcd598d5f4f19d94a97e6bbe7b8.svg?invert_in_darkmode&sanitize=true" align=middle width=157.72397519999998pt height=14.42921205pt/></p>

* the first equation with Δ is about displacement. the second with dot is about velocities. both are related with the Jacobian
* the Jacobian relationship holds only for very small displacements d
* we cannot expect to do long distance moves using this relationship
* this is because to end up in the Jacobian we linearized the function f in the local point. in the small area around q. linearizations holds for small displacement dx
* as robot moves q changes so the Jacobian does not hold (maybe we can calculate a new one in RT)
* in practice this small delta
* the dimensionality of Jacobian also validates the matrix multiplication: dimensions of Δq (1 column by n rows) and Δx (1 column by m rows)

### 5.3 Jacobian Example: Planar 2-link Robot

* we use again the 2-link planar robot as an example
    * 2 revolute-joints q1 and q2 and 2 links of equal length  1m
* the FW Kinematics analysis is:
<p align="center"><img src="/tex/cba6ccfa0418550b1674e316ab7d7150.svg?invert_in_darkmode&sanitize=true" align=middle width=219.07084979999996pt height=59.1786591pt/></p>
<p align="center"><img src="/tex/d54c10b08ab2addae271f2caf39738fa.svg?invert_in_darkmode&sanitize=true" align=middle width=89.02755674999999pt height=23.5253469pt/></p>
<p align="center"><img src="/tex/d06ab10497a349344faf52538b8005ea.svg?invert_in_darkmode&sanitize=true" align=middle width=98.94198929999999pt height=23.5253469pt/></p>
<p align="center"><img src="/tex/1f156eac0de2b7034fcf7e59b68294f8.svg?invert_in_darkmode&sanitize=true" align=middle width=61.84355265pt height=16.438356pt/></p>
<p align="center"><img src="/tex/6e0047d76949136b987d2afcfdb6b279.svg?invert_in_darkmode&sanitize=true" align=middle width=229.2863232pt height=39.452455349999994pt/></p>

* we can now calculate the jacobian
<p align="center"><img src="/tex/139ac2cd43537d9b961dcf5475c1ea87.svg?invert_in_darkmode&sanitize=true" align=middle width=435.9569643pt height=49.315569599999996pt/></p>

* in shortform Jacobian is
<p align="center"><img src="/tex/3ede65ac6506447e7d9675a9607cb874.svg?invert_in_darkmode&sanitize=true" align=middle width=171.25570439999998pt height=39.452455349999994pt/></p>

* we can now calculate the Jacobian for a particular spot     <img src="/tex/f4f87732a74fd904b8ff29c9f2fb5e1a.svg?invert_in_darkmode&sanitize=true" align=middle width=106.39222604999999pt height=35.5436301pt/>
* we draw the pose of the robot arm and calculate the Jacobian
<p align="center"><img src="/tex/f000e8b38bca2a5077f56f63bda0929b.svg?invert_in_darkmode&sanitize=true" align=middle width=283.1469729pt height=39.452455349999994pt/></p>=<p align="center"><img src="/tex/e7392c45a9b382e458642272b00c8112.svg?invert_in_darkmode&sanitize=true" align=middle width=78.83126955pt height=49.315569599999996pt/></p>=\frac{\sqrt{2}}{2}<p align="center"><img src="/tex/b76c12fca5cc33d5816604e7c306bab0.svg?invert_in_darkmode&sanitize=true" align=middle width=50.2284453pt height=39.452455349999994pt/></p>

* so we have the jacobian for this position... we can use the Δ equation and given the Δx calc the Δq needed <img src="/tex/df01ad8f7c411cbab4b6cb1803c84f97.svg?invert_in_darkmode&sanitize=true" align=middle width=106.85483984999999pt height=26.76175259999998pt/>
* the jacobian inverse for this position is
<p align="center"><img src="/tex/90d46ab5eec427853d5ff893bdbea34e.svg?invert_in_darkmode&sanitize=true" align=middle width=373.19239484999997pt height=40.993000949999995pt/></p>

* it makes sense q1 became smaller and q2 larger. so its correct
* be careful with signs
* another example is if we want the end effector to move up
<p align="center"><img src="/tex/c0a685ffc00c8be66360a0a67283013f.svg?invert_in_darkmode&sanitize=true" align=middle width=204.82963709999999pt height=40.993000949999995pt/></p>

* this also makes sense. only q1 becames larger
* remember that we need to recompute jacobian for every position

### 5.4 Singularities

* we will try to calculate the Jacobian for the previous example when q2 = π.  it will be
<p align="center"><img src="/tex/d8248c8d28ca05b16ac7ba4e91a70ae9.svg?invert_in_darkmode&sanitize=true" align=middle width=101.89690829999999pt height=39.452455349999994pt/></p>

* this is an important case. robot arm has fully folded onto itself
* if i multiply the Jacobian with Δq=[Δq1,Δq2]T it doesnt matter what i change in q1. it wont have any effect in positon as 1st column of Jacobian is 0
* in the sketch we can verify that. if the robot rotates around q1 end effector is in 0,0 position
* in this situation q1 lost its ability to move the end effector
* this is a problem. another problem is that the determinant of the Jacobian is 0. we cannot invert it and we cannot compute joint val move if position changes (but it cant)
* if we try to compute Δq for an arbitrary Δx. the equation system that we have (see below) is unsolvable
<p align="center"><img src="/tex/52c7d96bcaea498d92db1fcd45e90fb1.svg?invert_in_darkmode&sanitize=true" align=middle width=218.6170272pt height=39.452455349999994pt/></p>

* movement is possible only if the derived equation holds. if not we cannot satisfy the equation
* for the specific robot config in this position the only movement possible is along the tangent of the circle arount the second joint if q2 changes.
* so we are locked. what happens if i am close to being locked. if q2 is not π but very close to it
* in that case in the jacobian instead of 0 we would have 2 very small vals ε1 and ε2. also the determinant of the Jacobian will be non-zero
* then we can attempt to solve the equation system and calculate the Jacobian inverse then calculate the Δq for a small Δx in this position
<p align="center"><img src="/tex/c972e078976def439af525aff1991d2a.svg?invert_in_darkmode&sanitize=true" align=middle width=134.40586664999998pt height=44.00564025pt/></p>
<p align="center"><img src="/tex/caebbeb299676eb811f92806dae03497.svg?invert_in_darkmode&sanitize=true" align=middle width=116.6034474pt height=36.09514755pt/></p>

* remember that ε is very small. what if we write a piece of SW that takes in Δx calculates jacobian inverse and Δq and sents it to the robot given that the robot is close to the q2=π position
* as we divide for ε Δq1 is huge and Δq is analogus to q dot aka speed. so robot will attempt to cover instanlty a huge distance
* this will destroy the robot!!!!!!!!!!
* so being close to a border position commanding the robot can cause instability
* this kind of position is called a Singularity. these positions occur when the determinant of the Jacobian is 0
<p align="center"><img src="/tex/88cc0c4ac5412dbf86e97f6af28099d9.svg?invert_in_darkmode&sanitize=true" align=middle width=142.79527019999998pt height=16.438356pt/></p> 
* Being in a singularity means that a joint has lost its ability to move the robot
* Also being in a singularity measn we are constrained to move only in a specific direction only
* Being IN a singularity (locked) is better than being very close to a singularity. then asking for a finite movemtn can result in an infinite movement in joint space
* In a robot control software we must avoid singularities and approaching them
* this is done using a SW library that calculates the matrix condition (determinant). if its good then we are safe.
* if q2=0 (arm fully extent the jacobian is
<p align="center"><img src="/tex/0b892e88eec4c757909abb38bee3002c.svg?invert_in_darkmode&sanitize=true" align=middle width=130.35395505pt height=39.452455349999994pt/></p> 

* what we see is that columns are not lineraly indipendent. the determinant is 0. the robot is fully extent.
* if we move q1 the robot will move arount the max circle tangent at that position. same for q2. 
* the only possible movement is along the tangent line regardless of the joint angle changing val
* again we have the instability problem. so we lost the ability to move except on one line

### 5.5 Differential Kinematics Example- Spherical Robot

* we look at a 3D example the spherical robot.
* the position of the end effector in space is <img src="/tex/c86cbfd7b9c719f556118af4f1d6373b.svg?invert_in_darkmode&sanitize=true" align=middle width=80.95681934999999pt height=35.5436301pt/> as we dont care about the orientation
* the spherical robo has 3 DOF (3 Joints) so the q vecor is <img src="/tex/f9ff56915580f56ed56874f442f7e9dc.svg?invert_in_darkmode&sanitize=true" align=middle width=67.37066655pt height=35.5436301pt/>
* we recall the relationship of endeffector position with joint values
<p align="center"><img src="/tex/008a8c1c2f7c39478b1f9320cca01250.svg?invert_in_darkmode&sanitize=true" align=middle width=120.02097855pt height=59.1786591pt/></p>

* the general formula for the Jacobian is:
<p align="center"><img src="/tex/56d5580ac0191e8448b69c5c3db18db8.svg?invert_in_darkmode&sanitize=true" align=middle width=159.7490268pt height=69.04177335pt/></p>
* we compute the Jacobian for the general x case
<p align="center"><img src="/tex/1b1d37aef9e3531b696cc69fc6078847.svg?invert_in_darkmode&sanitize=true" align=middle width=304.1080746pt height=59.1786591pt/></p>

* we will now calculate the determinant of the Jacobian to be able to avoid the Singularities
<p align="center"><img src="/tex/af1c366dbb6b47f04e229c053f5a48ed.svg?invert_in_darkmode&sanitize=true" align=middle width=84.09242984999999pt height=18.312383099999998pt/></p>

* we solve for 0 to detect the singular positions to avoid <img src="/tex/fcd9beba58e5fb6f223a115cdfd81538.svg?invert_in_darkmode&sanitize=true" align=middle width=275.20608059999995pt height=24.65753399999998pt/>

* so its then the robot points up or down? why? because moving on q1 has no efect...(recall giant hops???)

* a common exercise when designing a robot is:
    * design the robot mech
    * get DH params
    * compute Forward kinematics
    * compute differential kinematics (Jacobian)
    * compute jacobian determinant to rule out singularities

### 5.6 Recap - Joint Space vs. Cartesian Space

* We need to understand the difference of Joint space vs Cartesian Space
* we drow the kinematic chain of a 3 joint robot arm (q1,q2,q3) with end effector in x position
* if there is a difference in the position o the effector by Δx. the resulting change in Joint angles is represented as Δq
* We formalize the notion of Joint Space (the space of possible joint values) and Cartesial(aka EndEffector aka Task) Space
* In Joint Space
    * we talk about joint values 
<p align="center"><img src="/tex/da0d1ff53357cedf33a54cb408332a3c.svg?invert_in_darkmode&sanitize=true" align=middle width=166.43640585pt height=22.7399535pt/></p>

* In Cartesian Space
    * we talk about relative position and orientation  of ee regarding base frame (translation and rotation along each axis)
        * so we work in 6dimensional space if we care about position and orientation. if we care only about position in 3d space we work in 3d. if we cae about position&orientation in 2D plane we work in 3D
        * we call this space task space because it represents target position where the endeffector has to go
<p align="center"><img src="/tex/d32fcb0a2ae4eefc920017df9539f835.svg?invert_in_darkmode&sanitize=true" align=middle width=242.45246684999998pt height=22.7399535pt/></p>

* Everything we do about the analysis of robot arms has to do with moving between these 2 spaces.
* going from Joint Space => Cartesian Space is Forward Kinematics (FK)
* going from Cartesian Space => Joint Space is Inverse Kinematics (IK)
* when we do Differential Kinematics
    $\Delta q \overset{J}{\rightarrow} \Delta x$
    $\Delta x \overset{J^{-1}}{\rightarrow} \Delta q$
    $\dot{q} \overset{J}{\rightarrow} \dot{x}$
    $\dot{x} \overset{J^{-1}}{\rightarrow} \dot{q}$
* For singular points there will be some Δx for which i cannot compute any Δq as J-1 is not possible
* in mathematical terms
<p align="center"><img src="/tex/087c11bc59045406deba4bd8dab62ff4.svg?invert_in_darkmode&sanitize=true" align=middle width=255.91228575pt height=63.81200265pt/></p>

* also:
<p align="center"><img src="/tex/a12904891877503c864ba112d9419c16.svg?invert_in_darkmode&sanitize=true" align=middle width=77.3344011pt height=14.42921205pt/></p>
<p align="center"><img src="/tex/edbbaa273a225a0c940bcdd4a8aa469d.svg?invert_in_darkmode&sanitize=true" align=middle width=49.93703385pt height=14.42921205pt/></p>

### 5.7 Differential Kinematics Example

* 1 more full example of Differential Kinematics
* The DH params of the robot are
    * Joint1: θ1=q1, d1=0, a1=l1, α1=90deg
    * Joint2: θ2=q2, d2=0, a2=0, α2=90deg
    * Joint3: θ3=0, d3=q3, a3=0, α3=0
* We will draw the robot, do FW kinematic analisys, differential kinematic analysis, detect the singular configurations
* we sketch the robot:
    * start from base frame (x to viewer, z up, y right)
    * first joint is rotating around z by θ1. link translated on x by l1 a fixed system param (not a degree of freedom). at the end of the link fram rotates on x by 90deg so z will point left and y up
    * second joint is a rotation around new z by θ2 and frame is rotated on x axis by 90deg. so new frame z point down and y left
    * last joint is prismatic so there is a variable translation on new z axis (pointing down) by q3
* robot is like a 2link planar robot rotating around base z axis with a variable length second link also first q on base is fixed (rotating around z on x,y plane)
* we go through joints doing forward kinemativc analysis to get the base to end effector transform matrix
<p align="center"><img src="/tex/2d40c90df8ba361ff4f11ce1d999a838.svg?invert_in_darkmode&sanitize=true" align=middle width=790.7995160999999pt height=78.9048876pt/></p>

* remember teh multiplication and trick (trnaslate+rotation) and get the final transform matrix
* also if we say we care only for the translation part of forward kinematics our equations are simplified. we keep only the transaltion part of last matrix replacing it with a [0 0 3 1]T vector
<p align="center"><img src="/tex/24e79a36e6d5e7e14e56b34fa6c1335b.svg?invert_in_darkmode&sanitize=true" align=middle width=167.66439195pt height=78.9048876pt/></p>

* we do a sanity check projecting the end effector point from the sketch along the 3 axis
* we care only about position so x will be x=[x y z]T
* we calculate the Jacobian
<p align="center"><img src="/tex/66ca001a3411eedb771febcf259a9217.svg?invert_in_darkmode&sanitize=true" align=middle width=413.71606814999996pt height=69.04177335pt/></p>

* we calculate its determinant (choose the easiest row to expand, add signs to heve the sign, multiply) remember s2+c2=1
<p align="center"><img src="/tex/c36ef00417bc05b472e3e1381aca0702.svg?invert_in_darkmode&sanitize=true" align=middle width=932.5308597pt height=18.312383099999998pt/></p>

* this robot is in singular position when |J|=0 so a) q3=0 because q2 becomes irrelevant b) when s2q3+l1=0 when end effector touches the z axis so q1 becomes irrelevant

### 5.8 Complete Kinematic Analysis Example

* we consider another robot to showcase full kinematic analysis complete (with position and orientation)
* The DH params of the robot are
    * Joint1: θ1=0, d1=q1, a1=0, α1=0
    * Joint2: θ2=q2, d2=0, a2=0, α2=-90deg
    * Joint3: θ3=0, d3=q3, a3=0, α3=0
* we draw the robot with base frame z->up, x->viewer, y-> right
* joint1 is prismatic so it translates the base frame by q1 on the z axis. the new frame after translated is not rotated
* joint2 is revolute it rotates around z by θ2. no translation but new frame is rotated on x by -90deg so new z points right and y down
* joint3 is prismatic extending on the new horizontal z axis by q3. new coordinate frame in nd efector is unchanged
* robot resembles a gamma of extensible links rotating on z
* we do the FK analysis (in the last 2 matrices rotation is multiplied with identity so stays unchanged, translation gets rotated by rotation matrix)
<p align="center"><img src="/tex/4c6999ad53d58e4bc37d6034a5289238.svg?invert_in_darkmode&sanitize=true" align=middle width=1020.5007994499999pt height=78.9048876pt/></p>

* we do the IK analysis for position only assuming we want our end effector to end up in position [a,b,c]T
* the translation part of the Transform matrix has to put the end effector in the position we want so we have our equation system
<p align="center"><img src="/tex/a5f4bb873ae7bdd458559608c0316a2c.svg?invert_in_darkmode&sanitize=true" align=middle width=87.79682174999999pt height=59.178683850000006pt/></p>

* c we have. its q1 for the other two we square them up and add them το get <img src="/tex/a8d0105699db1153770489d226fd930d.svg?invert_in_darkmode&sanitize=true" align=middle width=113.69852339999997pt height=28.712280299999996pt/>
* we start investigating solutions:
* if <img src="/tex/1a965ecd3cc084459ec719dc7e1c03e3.svg?invert_in_darkmode&sanitize=true" align=middle width=80.72090234999999pt height=26.76175259999998pt/> we have one solution for q3=0
* if <img src="/tex/f8565040d2ca6e3f96659e2935269e67.svg?invert_in_darkmode&sanitize=true" align=middle width=80.72090234999999pt height=26.76175259999998pt/> we have 2 solutions for q3 so we use atan2 so for q2
<p align="center"><img src="/tex/15fda88944475815aab7207fbf8b9cb1.svg?invert_in_darkmode&sanitize=true" align=middle width=306.64356195pt height=37.0084374pt/></p>

* we do differential kinematics analysis calculating the Jacobian
<p align="center"><img src="/tex/b5a24738d02bd95c166869c0ada93df3.svg?invert_in_darkmode&sanitize=true" align=middle width=166.07125754999998pt height=59.1786591pt/></p>

* the determinant of the Jacobian is
<p align="center"><img src="/tex/e2b8b4cb28cf7e4e2159787033af4f6b.svg?invert_in_darkmode&sanitize=true" align=middle width=182.2107177pt height=18.312383099999998pt/></p>

* so robot is in singular posisiton when q3=0 which is logical as rotation q2 has no effect

### Assignment 2

* Consider the robot described by the D-H table below:
    * Joint1: θ1=q1, d1=0, a1=2, α1=90deg
    * Joint2: θ2=q2, d2=0, a2=0, α2=90deg
    * Joint3: θ3=0, d3=q3, a3=0, α3=0
* Consider three possible robot sketches, one of which is a correct representation of the robot defined in this problem:
**Sketch 1:**

![sketch1](http://roam.me.columbia.edu/files/seasroamlab/imagecache/103x_h1_r1c.jpg)

**Sketch 2:**

![sketch2](http://roam.me.columbia.edu/files/seasroamlab/imagecache/103x_h1_r1a.jpg)

**Sketch 3:**

![sketch 3:](http://roam.me.columbia.edu/files/seasroamlab/imagecache/103x_h1_r1b.jpg)

* Compute the translation part of the Forward Kinematics transform <img src="/tex/23c72e543e8cfe1d17213857eef3d06f.svg?invert_in_darkmode&sanitize=true" align=middle width=33.38576999999999pt height=27.91243950000002pt/> from the base of the robot to the end-effector. In other words, derive the expressions for the components of the  <img src="/tex/381872314d30ee5ea556f738a0875a68.svg?invert_in_darkmode&sanitize=true" align=middle width=36.52961069999999pt height=21.18721440000001pt/> vector  <img src="/tex/78665e34641da3a3c7f74d8ea0e1f97b.svg?invert_in_darkmode&sanitize=true" align=middle width=7.35155849999999pt height=20.87411699999998pt/>  below:
<p align="center"><img src="/tex/3e73114122c1c5293f45c5e9676a08b7.svg?invert_in_darkmode&sanitize=true" align=middle width=182.63223825pt height=79.5616338pt/></p>

* We will use the notation from the lectures, i.e. 
<p align="center"><img src="/tex/b71330b956cb05d4b9559849628da9f9.svg?invert_in_darkmode&sanitize=true" align=middle width=92.7604062pt height=16.438356pt/></p>
<p align="center"><img src="/tex/2e9e9be025b9ee99a0e447e4cbc5f66a.svg?invert_in_darkmode&sanitize=true" align=middle width=134.1169104pt height=16.438356pt/></p>

* Compute the manipulator Jacobian with respect to end-effector position (and ignoring end-effector orientation). Find all the joint configurations where the Jacobian becomes singular.

## Week 6: Study Week

### Review and Practice Questions

* Welcome to Study Week. Please use this week to complete (or at least make significant progress on) currently released Projects. In the second half of the class we will be releasing two new Projects which will be of increased difficulty, so you will want to dedicate all available time to those.
* You can also review and recap the material from the first half of the class. Here are some example questions that can guide your review effort. These are also example questions you can use to prepare for the Final Exam, which will contain questions similar to (or selected from) these:
    * What are the conditions for a 3x3 matrix to represent a valid rotation in 3D space?
    * What are the conditions for a 4x4 matrix to represent a valid rigid body transform (expressed using homogenous coordinates) in 3D space?
    * In Denavit-Hartenberg notation, what are the four parameters that define a joint, and what does each of them mean?
    * If a robot arm is operating in 3D space, what is the smallest number of joints it must have in order to arbitrarily control both the end-effector position and orientation?
    * If a robot arm is operating in 3D space, and must arbitrarily control both the end-effector position and orientation, what is the smallest number of joints it must have in order to have a redundancy in any configuration? 
    * How do you define the joint space of a robot? What is the dimensionality of a robot's joint space?
    * Given a point in a robot's joint space, how do you find its corresponding end-effector position, expressed in Cartesian space?
    * Given a set of joint velocities, expressed in joint space, how do you find its corresponding end-effector velocity, expressed in Cartesian space?
    * Given an end-effector velocity expressed in Cartesian space, how do you find its corresponding set of joint velocities, expressed in joint space?
    * What are the matrix dimensions of a robot's Jacobian, assuming the robot operates in 3D space and controls both end-effector position and orientation?
    * How can we tell that a robot arm is in a singular configuration?
    * What are the practical implications of a robot arm being in a singular configuration?
    * How can we tell that a robot arm is approaching (but is not yet exactly in) a singular configuration?

## Week 7: Robot Arms - Cartesian Control

### 7.1 Problem Statement

* we will apply differential control and apply the theory to the problem of Cartesian Control a common problem in robotics
* We assume we have a robot with a known kinematic chain and that we have done forward kinematics.
    * we have the base->endeffector transform bTee_current
    * we have the intermendiate joints cooerdinate frames
* We want the end effector to move to a new position for which we know the bTee_desired
* We also want the end effector to move in a straight line
* What happens is a change in the cartesian pose of the end effector Δx
* If the Δx is given to us by the user, the operator of the we want the robot to execute it
* Usually we will have a robot that accepts velocity commands. in that case we need to compute a set of velocities that we send to the joints
* If we get Δx by the user we have to convert it to velocity in the direction of the Δx <img src="/tex/15483a62bcadc667bb9f7409dcb4c6fb.svg?invert_in_darkmode&sanitize=true" align=middle width=62.90514119999998pt height=22.465723500000017pt/> where ρ is the proportional gain. the equation is a proportional controller. from this cartesian velocity we will compute the velocity to send to the joints using the differential analysis equation and the jacobian <img src="/tex/930fad86ad58b649160a8607a6706224.svg?invert_in_darkmode&sanitize=true" align=middle width=49.93703384999999pt height=22.465723500000017pt/> this q dot is what we send to the robot
* this works in posiiton and orientation
* in our case where the Jacobian comes from. in our lectures we started with forward kinematic analysis. we had <img src="/tex/d828cc7d0493f4978d93b1fcf437d661.svg?invert_in_darkmode&sanitize=true" align=middle width=61.843552649999985pt height=24.65753399999998pt/> as result of FWD Kinematics. If we have the analytical function of FW kinematics we just have to calculate partial derivatives and build the Jacobian matrix <img src="/tex/aed03e86fea5e99701c2565812dd04ee.svg?invert_in_darkmode&sanitize=true" align=middle width=50.01419774999999pt height=30.648287999999997pt/>
* x is a 6D vector (position,orientation)
* this is complex.
* what we do for a robot where we dont have the analytical function. then we can compute the Jacobian numerically

### 7.2 Numerical Jacobian Computation

* we start by looking at a single robot joint. we dont have the analytical functions but we have solved the FWD Kinematics as we know the Transorms for all the coordinate frames including the one we have under investigation in joint j
* When we worked with URDF and computed FWD Kin publishing to the TF without having solved the analytical methods. we have bTj for joints up to bTee
* what havens when j moves say turns around the local z-axis. what happens then to the end effector??
* say if the joint rotates what a given velocity what is the velocity to the end effector?
* we dont want to compute the velocity of ee to the joint coordinate frame but in its own coordinate frame. 
* so if Vj whats the Vee ? 
* the trick is to consider the rest of the robot rigid
* The simplified problem is:
* Assume a rigid robot body except from the joint A with coordinate frame A
* end effector B has its own coordinate frame B
* The velocity of joint A expressed in coordinate frame A is: 
<p align="center"><img src="/tex/86edbeea685b22c9cdf5d345b37c6e51.svg?invert_in_darkmode&sanitize=true" align=middle width=132.68632739999998pt height=118.35734295pt/></p>

* ω is the angular speed of joints
* we want to know what is the resulting velocity of B in coordinate frame B: <img src="/tex/a0e25d7304db117e8901a32a931b57ff.svg?invert_in_darkmode&sanitize=true" align=middle width=57.33220019999999pt height=27.6567522pt/>
* We know the transform from A to B is:
<p align="center"><img src="/tex/561581e699eedb798f9ba84c5c31fc92.svg?invert_in_darkmode&sanitize=true" align=middle width=149.79836685pt height=39.4623702pt/></p>

* The velocity of B is a 6by6 matrix:
<p align="center"><img src="/tex/a0a5606a639ce2fc89724501d9bef7a1.svg?invert_in_darkmode&sanitize=true" align=middle width=279.0867618pt height=39.4724781pt/></p>

* what we understand is that the angular velociity of B will be the angular velocity of A rotated by <img src="/tex/667de7dfcf380344bfb6dba0a4442059.svg?invert_in_darkmode&sanitize=true" align=middle width=33.681900449999986pt height=27.6567522pt/> which is the transpose of <img src="/tex/6b5e0989fbafb974a165ff3d8f307778.svg?invert_in_darkmode&sanitize=true" align=middle width=33.681923549999986pt height=27.6567522pt/>
* posiional (translation) part of velocities plays no part in angular velocity
* the translation part of the velocity conversion has a rotation part of the translational velocity
* also the translational velocity has to do with the rigid body which is as well rotated..
* matrix S is a skeyw-symmetrix matrix: 
<p align="center"><img src="/tex/92adfb087559742698fb1440019fb56c.svg?invert_in_darkmode&sanitize=true" align=middle width=238.2802983pt height=59.1786591pt/></p>

* S matrix has the nice property of <img src="/tex/3db2bb7fbaba75628aa2eb60e2539597.svg?invert_in_darkmode&sanitize=true" align=middle width=109.18152134999997pt height=24.65753399999998pt/> so an easy way to express a cross product as matrix multiplication
* so  the upper right element is the cross product of the model arm with the rotation of frame from joint to end effector which sets the translational velocity conversion with the pure rotation
* the trransform matrix producing the rotation matrix and translation matrix we use for the velocity conversion we get from FWD Kinematic analysis
* for conversion from joint j to end effector ee
<p align="center"><img src="/tex/f22e01bb160b93083ee8c51e7de45ba4.svg?invert_in_darkmode&sanitize=true" align=middle width=215.92878614999998pt height=39.452455349999994pt/></p>
<p align="center"><img src="/tex/0c7a8e44320c6cf005bffbb6dee6370f.svg?invert_in_darkmode&sanitize=true" align=middle width=85.64142344999999pt height=15.936036599999998pt/></p>

* the velocity of joint j can be of anytype. but assuming its a revolute joint we say th ony possible velocity is a rotation around z so
<p align="center"><img src="/tex/2b83e080574fd227c23eb47661b53ef7.svg?invert_in_darkmode&sanitize=true" align=middle width=193.63080165pt height=17.031940199999998pt/></p>

* in this case only the last 6th column of Vj matters [:,5]in calculating ee velocity which will be 6by1 vector
* robot have many joints. what if multiple joints move simultaneously. assuming all arrre revolute around z::
<p align="center"><img src="/tex/c1c58c6e77623f4058db5acce8b31921.svg?invert_in_darkmode&sanitize=true" align=middle width=371.4478515pt height=16.438356pt/></p>
<p align="center"><img src="/tex/482fb4af4410bc16194b810256ae9044.svg?invert_in_darkmode&sanitize=true" align=middle width=127.14845549999998pt height=47.35857885pt/></p> 

* this can be presented in matrix multiplication form wher V[:,5] is a 1xn vector multiplined with dotq a nx1 vector 
* we express the relation witht the jacobian notion where
<p align="center"><img src="/tex/c5844e6fc239935f68ea7b453f2a2797.svg?invert_in_darkmode&sanitize=true" align=middle width=345.37923255pt height=78.9048876pt/></p>
<p align="center"><img src="/tex/650af9dbe4766b12c7ad3faf107ae367.svg?invert_in_darkmode&sanitize=true" align=middle width=75.2988357pt height=14.42921205pt/></p>

* this is the Numerical Jacobian. 
* Knowing only the relative transforms from joint coordinate frames to end-effector coordinate frame we build the transmission matrices V
* Vee is expressed in end effector coordinate frame
* Δx we were given is  in relation to base coordinate frame
* if we have the <img src="/tex/ba321f9d63bd563265e21e624b49821f.svg?invert_in_darkmode&sanitize=true" align=middle width=143.42674829999999pt height=47.81931659999997pt/>
* then what we have as Velocity of the end effector exrpessed in its own coordinate frame related to the speed of x relted to the base frame <img src="/tex/0034def504d242026a8951e69002874e.svg?invert_in_darkmode&sanitize=true" align=middle width=74.77712219999998pt height=22.465723500000017pt/>is
<p align="center"><img src="/tex/70c8246a68c452bc0ec02e4cbe4c58d1.svg?invert_in_darkmode&sanitize=true" align=middle width=164.61923775pt height=39.452455349999994pt/></p>

* where the see that the speed on the base frame isthe end effecrtor frame velocity rotated
* so the way to work is we are given Δx => xdot => Vee => calculate the Jacobian => get qdot

### 7.3 Singularity Avoidance: Jacobian Pseudoinverse

* we recap:
<p align="center"><img src="/tex/4c8dbb63c29c2872a058fa77f4e77c90.svg?invert_in_darkmode&sanitize=true" align=middle width=76.7160339pt height=14.42921205pt/></p>
<p align="center"><img src="/tex/cc02ee75e875f280b60b798144b2a072.svg?invert_in_darkmode&sanitize=true" align=middle width=72.72456179999999pt height=14.553275549999999pt/></p>

* n is the number of robot joints and m is the number of variables we are controlling in end effector
* we compute qdot
<p align="center"><img src="/tex/c09cd52678f4b63d14f03f01ec3428c4.svg?invert_in_darkmode&sanitize=true" align=middle width=80.253393pt height=17.399144399999997pt/></p>

* To have an inverse of a Jacobian matrix it must have equal dimensions m=n and Jacobian must be a full rank
<p align="center"><img src="/tex/87c66a7bf1d518e0c967b85a3bbb7a97.svg?invert_in_darkmode&sanitize=true" align=middle width=103.88521109999999pt height=17.399144399999997pt/></p> 

* the above shows that J-1 must be at least the Right side Inverse of J so the constraint is m<=n but Jacobian still has to be full rank
* The problem is that as Jacobian approaches the singularity <img src="/tex/6bb7849c352a67a42adf9f3546695a39.svg?invert_in_darkmode&sanitize=true" align=middle width=49.93706849999999pt height=21.95701200000001pt/> so again we send infinite velocities to the robot
* We will use linear algrbra and the Singular Value Decomposition of a Matrix
* we write the jacobian as the product of 3 matrices <img src="/tex/f6cd8bfdab77f7f21ae09716aa555e91.svg?invert_in_darkmode&sanitize=true" align=middle width=55.16363984999999pt height=27.6567522pt/> where J is m x n
* <img src="/tex/e2d9a2290b248860f8a59b420e03e59d.svg?invert_in_darkmode&sanitize=true" align=middle width=150.67895205pt height=27.6567522pt/> so U is square and orthogonal
* <img src="/tex/74de91321f0ecddc7feb69b2b4d3fe94.svg?invert_in_darkmode&sanitize=true" align=middle width=73.90039139999999pt height=26.17730939999998pt/> Σ is diagonal matrix
* <img src="/tex/83aff904452a24b41bd49209a0c4f1c7.svg?invert_in_darkmode&sanitize=true" align=middle width=144.27957224999997pt height=27.6567522pt/> so V is square and orthogonal
* if m<=n:
<p align="center"><img src="/tex/e7a6be565d8a758178e98d3c0735dc7e.svg?invert_in_darkmode&sanitize=true" align=middle width=219.4693677pt height=59.1786591pt/></p>

* the sigma values on the diagonal ara on descending order so σ1 >= σ2 >= ... >= σm >= 0
* also if n = RANK(J) all values past it will be 0 <img src="/tex/42fb091faa642fd24257731b1599afc8.svg?invert_in_darkmode&sanitize=true" align=middle width=91.58292494999999pt height=22.831056599999986pt/>
* if the Jacobian is Rank defective so its rank is m-1 σm = 0. 
* this is a way to tell by eye the rank of a matrix 
* a robust way to tell numerically that a matrix is approaching the singularity, being close to lose rank iswhen: <img src="/tex/0ba17240bea9b2d71e4f9f9fc7cf6a7a.svg?invert_in_darkmode&sanitize=true" align=middle width=82.88516114999999pt height=23.388043799999995pt/>
* the program might decide when seeing is close to lose rank aka approaching singularity to stop moving to avoid issuing infinite command for protection. this is not optimal as the robot is stuck.
* the correct apporach is to allow to go back but not towards the singularity
* we will see another better way..
* we ll see the matrix we get when we invert Σ
<p align="center"><img src="/tex/7dc9dc69e06add2ad0ac58083d201a2e.svg?invert_in_darkmode&sanitize=true" align=middle width=160.38770549999998pt height=80.37874679999999pt/></p>
<p align="center"><img src="/tex/143ce1f516754f1bd4dc17f2fe33f793.svg?invert_in_darkmode&sanitize=true" align=middle width=117.21669630000001pt height=17.399144399999997pt/></p>
<p align="center"><img src="/tex/7c50fb65cb78125e71347e095ae37992.svg?invert_in_darkmode&sanitize=true" align=middle width=127.4467293pt height=14.6502939pt/></p>

* the last equation is fine when Jacobian holds rank. as it starts to lose rank the 1/σm gets bigger so the inverse Jacobian gets bigger. when σm is 0 we cannot even invert the Jacobian as we get infinity
* the cheap trick is when we see that <img src="/tex/2386edbf7233cbe407630abdae00353f.svg?invert_in_darkmode&sanitize=true" align=middle width=57.31456005pt height=24.65753399999998pt/> then in the position of <img src="/tex/d5bbe85db313fad95e99d30a823054c1.svg?invert_in_darkmode&sanitize=true" align=middle width=18.5327307pt height=27.77565449999998pt/> we put 0 in the Σ-1. all the οhter diagonal vals we invert normally and continue computation. but values that go to infinity we replace them with 0
* then we call the matrix
<p align="center"><img src="/tex/1d61af7d8f514bc0877bf765a18bf976.svg?invert_in_darkmode&sanitize=true" align=middle width=21.9635526pt height=13.910572499999999pt/></p>
<p align="center"><img src="/tex/f2e904eacb413f2fb525a92048b8b05c.svg?invert_in_darkmode&sanitize=true" align=middle width=113.97636524999999pt height=14.6502939pt/></p>

* J+ is a very importan matrix. its called the Jacobian pseudo inverse. it has some very important properties for us
* If J is a full rank Jacobian <img src="/tex/5b0ced10e36027c37081514c7c957b48.svg?invert_in_darkmode&sanitize=true" align=middle width=59.886844049999986pt height=26.17730939999998pt/>
* If J is a low rank Jacobian <img src="/tex/94d9952e005596a4920bb941eefc98cb.svg?invert_in_darkmode&sanitize=true" align=middle width=59.886844049999986pt height=26.17730939999998pt/>
* If we compute qdot with J+ <img src="/tex/e382936fc52d5059098575c330e3f16e.svg?invert_in_darkmode&sanitize=true" align=middle width=75.75733934999998pt height=26.17730939999998pt/> we get some excellent properties:
    * if J is full rank it is an exact solution of $$\dot{q}=V_ee$
    * if J is low rank the angular velocities computed wont allow any additional movement towards the singularity but will allow any movement that does not get us closer to the singularity. SWEETT!! we get the best of both worlds
* In Practice any linear algebra lib has methods to compute the pseudo inverse `numpy.linalg.pinv(J,epsilon)`

### 7.4 Putting It All Together: Cartesian Control

* How we do Cartesian Control?
**Given:**
* the coordinate frames for all the joints: <img src="/tex/e42e1d4e3b618997f53b1fed51eeae4a.svg?invert_in_darkmode&sanitize=true" align=middle width=24.148830749999988pt height=24.65753399999998pt/>
* the transform matrix from the base to the current position of the end effector <img src="/tex/0f5ffa7ee60ec93ab01f49f97987a192.svg?invert_in_darkmode&sanitize=true" align=middle width=74.85815864999998pt height=27.91243950000002pt/>
* the transform matrix from the base to the desired position of the end effector <img src="/tex/324e8766b8bd45538586c8d04030dd42.svg?invert_in_darkmode&sanitize=true" align=middle width=72.21403859999998pt height=27.91243950000002pt/>
**Assume:**
* all joints are revolute
**Output**
* Joint angle velocity <img src="/tex/84af046067aaaa42645f613d1c5925bd.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928106449999989pt height=21.95701200000001pt/>
**Steps**
* from the 2 base to end-effector Transform matrices we compute a Δx: <img src="/tex/a27c7afa3bd139c03e949be2623df7c1.svg?invert_in_darkmode&sanitize=true" align=middle width=154.71156689999998pt height=27.91243950000002pt/>
* from the Δx we get xdot multiplying with the gain, the gain is set by trial and error (proportional control) <img src="/tex/15483a62bcadc667bb9f7409dcb4c6fb.svg?invert_in_darkmode&sanitize=true" align=middle width=62.90514119999998pt height=22.465723500000017pt/>
* using the transform matrix we transform xdot to desired velocity of end effector in its own coordinate frame  <img src="/tex/ec55afaa615bc0b7e571f45bea1b87a5.svg?invert_in_darkmode&sanitize=true" align=middle width=93.83897159999998pt height=27.91243950000002pt/>
* for each joint j we compute matrix Vj relating velocity of joint at the joint coordinate frame to velocity of the ee expressed in its own coordinate frame. we keep only the last coolumn of this matrix (as we care for angular velocity in z) <img src="/tex/608d060ca3e7cdf000d08e34cca32ff5.svg?invert_in_darkmode&sanitize=true" align=middle width=342.1135839pt height=24.65753399999998pt/>
* We assemble these columns in block column form and get the jacobian <img src="/tex/d02fdb140459233b8b75946d58fb4bef.svg?invert_in_darkmode&sanitize=true" align=middle width=295.7797326pt height=24.65753399999998pt/>
* we  compute the pseudoinverse of J <img src="/tex/165206dbe1af27c37cd6db28900e3050.svg?invert_in_darkmode&sanitize=true" align=middle width=20.78772464999999pt height=26.17730939999998pt/> with ε
* we compute <img src="/tex/4c1b70ed304f6fcbe7f6c94de7865767.svg?invert_in_darkmode&sanitize=true" align=middle width=73.5182217pt height=26.17730939999998pt/>
* we send <img src="/tex/84af046067aaaa42645f613d1c5925bd.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928106449999989pt height=21.95701200000001pt/> to the robot
* In practice we also put some safeguards along the computations. in robotics.. if something goes wrong the robot might cause an accident
**Safeguards**
* Scale <img src="/tex/c8a19da4c4ef5fa3fed69bb070243246.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=21.95701200000001pt/> such that <img src="/tex/e5fec28d237928190957cbaaeaa67f7e.svg?invert_in_darkmode&sanitize=true" align=middle width=41.06167559999999pt height=24.65753399999998pt/>  less than a threshold, to prevent sending too much movement to the robot
* Scale <img src="/tex/84af046067aaaa42645f613d1c5925bd.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928106449999989pt height=21.95701200000001pt/> such that <img src="/tex/5e7d7ed3b501049141dcaf64febe2e79.svg?invert_in_darkmode&sanitize=true" align=middle width=39.59477444999999pt height=24.65753399999998pt/>  less than a threshold, to prevent sending too much movement to the robot
* Alternatively  scale <img src="/tex/84af046067aaaa42645f613d1c5925bd.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928106449999989pt height=21.95701200000001pt/> such that <img src="/tex/f2ab2a27367e68f844720bd94c919d50.svg?invert_in_darkmode&sanitize=true" align=middle width=40.917318749999986pt height=22.831056599999986pt/>  less than a threshold, to prevent sending too much movement to the robot

### 7.5 Redundant Robots: Null Space Control

* Redundant robots are used mostly in research. Is one that has more than the smallest number of DOF needed to achieve any combination of position and orientation for end effector
* In 3D with position and orientation this means >6 DOF. it will give me more ways to achieve the same orientation and position (infinite in some configs)
* we write our main equation <img src="/tex/0ed1200da538521f93ee7c64dee2482b.svg?invert_in_darkmode&sanitize=true" align=middle width=64.8440529pt height=22.465723500000017pt/>
* we want to know if thre are angular velocities different than 0 so that when multiplied with J they are 0 <img src="/tex/7fd2d3f8d76c2707d2db4c6dfa91c454.svg?invert_in_darkmode&sanitize=true" align=middle width=187.36670699999996pt height=22.831056599999986pt/>
* what we are really asking is if there are joint velocities that produce no velocity to the end effector <img src="/tex/c3b613ed28a01236e980201dd1e4ee1b.svg?invert_in_darkmode&sanitize=true" align=middle width=57.11941949999999pt height=22.465723500000017pt/> means that dotqn is in the null space of the Jacobian
* if m >= n the above happens only at singularities
* what is the dimensionality of the null space of the Jacobian: at least 1
* if m < n (if we have more joints than we need) the lin algebra rank-nullity theorem tells us that thats always the case. it means that at any moment we can move the joints in a way that does not produce movement to end effector
* the way to compute the qdot in the null space of the Jacobian is by projecting the  input into the null space. 
* to do it if we have any joint velocity q dot we left multiply it with 1-J+J then its guaranteed to always be in the null space of the Jacobian
<p align="center"><img src="/tex/b7ba36fc0256c6657bea913b0067e2c8.svg?invert_in_darkmode&sanitize=true" align=middle width=285.95199105pt height=18.0201615pt/></p>

* this is because <img src="/tex/72e4d269c8070140ec92633716b33cef.svg?invert_in_darkmode&sanitize=true" align=middle width=75.61632209999999pt height=26.17730939999998pt/> which holds always: whether J is full rank or not. 
* if J is full column rank: <img src="/tex/5b0ced10e36027c37081514c7c957b48.svg?invert_in_darkmode&sanitize=true" align=middle width=59.886844049999986pt height=26.17730939999998pt/>. 
* if J is full row rank: <img src="/tex/536ae59b14ab10b3cc00b80849f1db57.svg?invert_in_darkmode&sanitize=true" align=middle width=59.886844049999986pt height=26.17730939999998pt/>
* if i  calculate the Jacobian pseudoinverse and the the solution for dotq <img src="/tex/a3ecbe792ca8bf50f4637a1df6a6e809.svg?invert_in_darkmode&sanitize=true" align=middle width=82.19384744999998pt height=26.17730939999998pt/>
* before sending it to robot we can have another goal for the end effector. to move the joints without moving the end effector 
* then what I will send to the robot will be <img src="/tex/f82fe3ea1905b5493ff4a2e4f9c5efc9.svg?invert_in_darkmode&sanitize=true" align=middle width=680.9945373pt height=26.17730939999998pt/>\dot{q}<img src="/tex/41e3f1e771d36cb3b976687e73f3038d.svg?invert_in_darkmode&sanitize=true" align=middle width=122.12099789999999pt height=22.831056599999986pt/>\dot{q_n}<img src="/tex/b90f33cb10b954e22a18a741b7e6da9e.svg?invert_in_darkmode&sanitize=true" align=middle width=3859.3526961000002pt height=5452.6027413pt/>q=[\frac{\pi}{3},\frac{\pi}{5}]<img src="/tex/4b415ebdb85703b577991c2fcb093a88.svg?invert_in_darkmode&sanitize=true" align=middle width=4299.32279865pt height=4819.5433869pt/>{(mass * (height * height + length * length) / 12)}"
                 iyy="${(mass * (width * width + length * length) / 12)}"
             izz="${(mass * (width * width + height * height) / 12)}"
             ixy="0" iyz="0" ixz="0"/>
  </macro>

  <!-- length is along the y-axis! -->
  <macro name="cylinder_inertia_def" params="radius length mass">
    <inertia ixx="${(mass * (3 * radius * radius + length * length) / 12)}"
                 iyy="${(mass * radius* radius / 2)}"
             izz="${(mass * (3 * radius * radius + length * length) / 12)}"
             ixy="0" iyz="0" ixz="0"/>
  </macro>

</robot>
```

## Week 9: Motion Planning II, Mobile Robots 

### 9.1 Preliminaries and Map Representations

* Most of the times a Mobile Robot operates in 2D space
* In low dimensional spaces we have multipme ways to represent the map (grid-discretized, polygonal-vertices)
* the map might come from floorplan or by measurements from sensors and built in realtime
* Remember that for 6D maps (mostly for robot arms) discretizing them is not feasible. for 2D is ok
* In Mobile robots maps the robot is not a point. it has dimensions. its not realistic to talk about point. so the algotithms that we have seen so far do not work per se
* a cheap trick to use them is to manipulate the maps (map inflation) by inflating the obstacles by the size of the robot. then we can use the point based algorithms

### 9.2 Motion Planning as Graph Search

* we look at a 2D map where the obstacles have been inflated so that the robot position can be treated as a point
* again we want to find the path between the start point and the goal point
* a common approach is to convert a map representation as a graph aka a collection of nodes and edges
* then we can use path planning algorithms that use graphs
* an easy way to convert a map to a graph is to produce a visibility graph. we connect any vertex of any obstacle and start.goal with any abailable vertex with a straight line as long as the line is unobstracted
* then we remove the obstacles from the picture. vertices are the graph nodes and edges are the lines
* if we travel start to end following the graph we will have no bumps to the obstacles
* we can assign costs to each edge depending on the cost of robot for moving through the edge. (e.g length or time)
* we aim the cost of the path to be as low as possible

### 9.3 Dijkstra's algorithm

* For any node n, keeps track of the *length of the shortest path from start to n found so far*, labeled g(n)
* Input: visibility graph, start S, goal G
* Output: path from S to G
* Algorithm:
    * Label all nodes 'unvisited'
    * Mark S as having g(S)=0
    * While unvisited nods remain:
        * choose unvisited node n with lowest g(n)
        * mark n as visited
        * for each neighbor r of n:
<p align="center"><img src="/tex/e67d77cd25ca15b1600f25c836a07631.svg?invert_in_darkmode&sanitize=true" align=middle width=224.0576283pt height=16.438356pt/></p>

* in this algo at anytime we keep track of the shortest path from start to a set of parrticular nodes
* all nodes are unvisited at first
* we know only start node and length to itself is 0: <img src="/tex/1752ec5c18fb5dee25a2eab9e0a5e912.svg?invert_in_darkmode&sanitize=true" align=middle width=62.38001384999998pt height=24.65753399999998pt/> we mark it as visited
* we look at S neighbours. the weight of the node is the node N1 is the g of S + the weight of the edge s->N1 so g(N1)=11 . similarly g(N2)=10. we mark N2 as visited because it has lowest weight. we look to all its unvisited neighbours N1,N3,N4 => g(N3)=21, g(N4)=42 g(N1)=28. because what we have for N1 is lower. lowest marked node is N1 with 11 so we mark it as visited
* we look at unvisited neighbors of N1 (N8,N3)
* we repeat till we visit the goal. th epath length is 58
* Dijksta's Algorithm
* for any node n, keeps track of the *length of the shortest path from start to n found so far*, labeled <img src="/tex/f010a0fda7cdcc04209d9381ef5fca27.svg?invert_in_darkmode&sanitize=true" align=middle width=31.08266699999999pt height=24.65753399999998pt/>
* Key Idea: visit closest nodes first
* Guarantee: once a node n has been "visited", <img src="/tex/f010a0fda7cdcc04209d9381ef5fca27.svg?invert_in_darkmode&sanitize=true" align=middle width=31.08266699999999pt height=24.65753399999998pt/> is equal to the length of the shortest part that exists from S to n.
* The algorithm is thus *guaranteed* to find the *shortest possible path* from S to G (along the graph)
* Running time: can be *quadratic* in the number of nodes
* When we write the q for a node we write from which node we are comming from. in this way we can extract the shortest path in the end easily

### 9.4 Graph Search on Grids

* we have seen how a polygonal map can be converted as a graph.
* what if our map comes as a grid.
* an easy way is to say that each (empty) cell (central point) is a node. also each cell is connected to its neighbors (like minesweeper). cost is higher for diagonals. ecqual for vertical or horizontal
* our graph will have many nodes
* we can apply Dikstas algorithm for grids. we get usually many cells with same weight along the way . not an issue we can choose randomly

### 9.5 A* search

* For any node n, also uses a *heuristic that estimates how far n is from the goal*, labeled here <img src="/tex/72b322da8035af6f39a0a9b5134877a2.svg?invert_in_darkmode&sanitize=true" align=middle width=32.12342429999999pt height=24.65753399999998pt/>
* A heuristic is *admissible* only if it never over-estimates the real distance. A commonly used hevristic that meets this requirement is straight-line distance to goal
* Input: visibility graph, start S, goal G
* Output: path from S to G
* Algorithm: 
* mark all nodes "unvisited"
* mark S as having <img src="/tex/d0f4ffa8f6381042610f47b8d9b94ecb.svg?invert_in_darkmode&sanitize=true" align=middle width=204.20824544999996pt height=24.65753399999998pt/>
* while unvisited nodes remain:
    * choose unvisited node n with lowest $f(n)$
    * mark n as visited
    * for each neighbor r of n:
<p align="center"><img src="/tex/e67d77cd25ca15b1600f25c836a07631.svg?invert_in_darkmode&sanitize=true" align=middle width=224.0576283pt height=16.438356pt/></p>
<p align="center"><img src="/tex/54c73ba921a34ed8def992267df9cc2f.svg?invert_in_darkmode&sanitize=true" align=middle width=131.70287295pt height=16.438356pt/></p>

* th intuition of this algo is to use the distance of a node to the goal. we dont know but we guess using a heuristic. and it should be optimistic. like straight distance to goal..
* so before even starting to iterate we have for all nodes (e.g grid cell centers) their h(n) filled with the distance from the goal in straight line. we ignore obstacles. of course we dont calculate for obstacles
* dijkstras algo explores first the node closer to the shortest path from the start
* A* explores first the node with higher chance to lead us to the goal faster
* A* reduces randomness and saves time
* we still use shortest path in our selection as f(n)=g(n)+h(n)
* remenber that g(n) is calculated with Dijkstra's algo logic
* we see A* going straight to goal then hitting the obstacle and backtracking and even improving the path
* A* in worst case is quadratic but in most cases is faster

### 9.6 Differential Drive Robots

* Αpplication to Real Robots

* so far, we have assumed that the robot can always *move in a straight line in any direction*
* that is a complex (and expensive) mechanism to realiz in practice. e.g we need fully rotating wheels
* a robot that has no constrains in velocity is referred as *holonomic*. it can generally move in any direction
* A common solution especially for indoor robots is: Differential Drive Robots
* 2 main drive wheels and a passive 3rd rotating that does no move
* the drive whwwls do not steer. but can be rotated with variable speed of drive wheels
* If the linear velocity of left wheel is VL and of the rigth wheel is VR,the distance between the wheels is l, the angular velocity of robot around a center of rotation with a radius distance from the projection of drive wheel position on the Drive wheel  inter distance is R, we have:
<p align="center"><img src="/tex/836b05aac0d7c330a5113d2c322410c9.svg?invert_in_darkmode&sanitize=true" align=middle width=99.11799314999999pt height=33.81208709999999pt/></p>
<p align="center"><img src="/tex/4c451908567f78a864a4d97318d1d986.svg?invert_in_darkmode&sanitize=true" align=middle width=98.17449839999999pt height=33.81208709999999pt/></p>
<p align="center"><img src="/tex/40d5ebf16fcb77ebc56d281c5ba66acf.svg?invert_in_darkmode&sanitize=true" align=middle width=120.42827609999999pt height=36.2778141pt/></p>
<p align="center"><img src="/tex/86025ed620c39a257b05205c40ad621c.svg?invert_in_darkmode&sanitize=true" align=middle width=94.6053306pt height=33.62942055pt/></p>

* if VR=VL then R=inf and \omega=0 robot is doing pure translation (no rotation)
* if VR=-VL the R=0 so robot turns in place
* any other combination os speeds has a translation part and a rotation part
* Differential Drive Pros:
    * only two powered wheels. both non-steered
    * no separate steering mechanism
* Differential Drive Cons:
    * cannot move "sideways" must turn and move (non holonomic)
    * passive caster wheel can still cause jerks
* Motion Planning for Differential Drive:
    * Robot often designed with circular foorprint
    * "Turn in place" almost as good as "drive in any direction" (it impersonates a holonomic robot so point algorithms apply)

### 9.7 Non-Holonomic robots

* We will now talk about oudoor robots. robots that do drive
* Car like steering is called Ackerman Steering, a common solution for outdoor robots
* all 4 wheels of the car move on cyrcles with the same center when steering
* wheels allways are on the tangents of these circles
* the center of the circles is  perpendicular to the back wheel axis
* Pros:
    * only two steered wheels
    * single steeering input
    * no sideways wheel slip
* Cons: 
    * in practice, turning radius cannot be arbitrary small
    * in particular cannot turn in place
    * non-holonomic, and cannot really approximate a holonomic robot
* Path Planning for Non-Holonomic Robots
    * orientation matters when planning
    * movement must be selected from allowed primitives
    * simple example: left,right,forward
    * must keep track of orientation
* when we do RRT for non-holonomic robots.
    * when we choose a node we cannot go in an arbitrary direction.
    * we chose one of the available discrete primitives (directions)
* other posibilities:
    * C-space extended to include orientation (x,y,θ)
    * or even derivatives

### 9.8 Recap

* Motion Planning for Mobile Robots:
* Search space is generally 2D or 3D (with orientation)
* Lends itself to discretization into grids, or polygonal obstacle representations
* obstacle images are often available
    * must  be inflated to accomodate robot size and allow use of algorithms for point robots
* motion planning formulated as *search on a graph*:
    * works on either polygonal or grid maps
    * algorithms: dijkstra, A* etc
* in real life:
    * indoors robots often use differential drive; with turn-in-place allows movement in any direction
    * outdoors robots (cars) can not move in arbitrary direction, so path planning must account for that.

## Week 10: Conclusion 

### 10.1 The Hall of Fame

* PUMA robot arm (1st wave of industrial robot)
    * pick and place: set position -> Inv Kin-> joint positions
    * welding: ee path => Cartesian Control => joint velocities
    * manual: user moves the robot
* The robot after getting the joint values
    * a controller runs a loop (PID). closed loop control
    * it sends current to motor
    * it uses an encoder to get joint value feedback
* Motor => moves Link => Link moves EE => an effect happed in physical world
* preprogrammed, precise, tireless job execution
* little feedback from environment

### 10.2 The Leading Edge

* New technologies post 2010 going in production
* Starting to integrate environment sensing
    * 3D scene geometry(machine vision,LIDAR,stereovision) eg point cloud
    * force information
    * touch information
* PR@ Robot has laser sensor and scans environment. it know sthing is there but does not know what it is
* semantic info is extracted with machine learning
* the data flow is 
<p align="center"><img src="/tex/072640841eeeef3872d4b3c737edd753.svg?invert_in_darkmode&sanitize=true" align=middle width=140.56237635pt height=15.936036599999998pt/></p>
<p align="center"><img src="/tex/1003d643d0d26e3b00be5d1a19551d18.svg?invert_in_darkmode&sanitize=true" align=middle width=367.0048008pt height=14.611878599999999pt/></p>

* Env Representation is built on the fly as new data come
* a way to store point cloud data is [octree](http://fab.cba.mit.edu/classes/S62.12/docs/Meagher_octree.pdf)

<p align="center"><img src="/tex/98b0910c24db4cf27c9d9ac1a77658a7.svg?invert_in_darkmode&sanitize=true" align=middle width=680.9028880499999pt height=17.031940199999998pt/></p>

* motion planner produces the path
* trajectory generator adds velocity, accelaration time to the path, making it smooth to execute without stoppung, making sure robot does not wxceed its limits
* result goes to motion control a closed loop control of the motor sending current and getting encoder info (PID position controller)
* motor produces torque and through a gearbox it goes to joint moving the link hich applies force to the world
* sensing of change in env happens ~30Hz
* a good motion planner produces output at ~5Hz
* PID control has a ~1kHz freq
* torque measurement is new technique: torque encoder in motor and encder in joint attached to a string in gearbox. based on the difference of these encoders we measure torque we know the torque
* this is called series elastic actuation. we see a torque spike that goes to motion planner
* in some robots links are equiped with touch sensors . info goes to env. representatio
* Papers on subject:
* [Perception, Planning and Execution for Mobile Manipulation in Unstructured Environments](https://pdfs.semanticscholar.org/c4e2/88f3bf1b53c85e4bbf93edd6ed2affec7105.pdf)
    * The sensing Pipeline. Filtering 3D sensor data to remove noise and points corresponding to robot parts
    * Environment Modeling: octree-based represenation used to generate occupancy grid model
    * Segmentation and Recognition: 1) support surfaces and objects are segmented from 3D data 2) known objects (based on object detection) are represented using meshes 3) other objects are represented using point clusters
    * Grasping. cluster planner plans grasps for objects represented as point clusters
* Intelligent robot operating on its own

### 10.3 The Future

* Key fileds needed for robots to make the next big leaps forward
* Semantic Perception
    * can it be separated from hardware?
    * continuum: from segmentation to recognition
    * "smaller" domain helps (e.g roads)
    * machine learning making great strides
* Reasoning and Planning under uncertainty
    * task planning to motion planning
* Complex reactive motor skills
    * manipulation, legged locomotion, etc... 
* Papers:
    * [Colaborative Grasp Planning with Multiple Object Representations](https://roam.me.columbia.edu/files/seasroamlab/publications/icra2011_collaborative.pdf) 
* Thoughts:
    * thing about outliners
    * object recognition DB
    * modeling of everything is required... intuition??
* Past: Structured env
* Present: semi-structured env
* Future: fully unstructured env

### Review and Practice Questions

* This week concludes the lectures in this course. You can use this week to complete the final projects, recap class material, and prepare for the final quiz.
* Final Quiz Info: the Final Quiz will contain 10 questions and you will have 1 hour allotted time for completion.
* Here are some example questions covering material from the second half of the course that can guide your review effort. These are also example questions you can use to prepare for the Final Exam, which will contain questions similar to (or selected from) these.
* (Reminder: practice questions for the material covered in the first half of the class are available in Week 6. The Final Quiz will cover all the material in the course).
    * What are the differences between a matrix inverse and a pseudo-inverse?
    * When is it impossible to compute the inverse of a robot Jacobian?
    * How can you avoid coming close to singular robot configurations when doing Cartesian Control?
    * When performing Cartesian Control in order to get the end-effector from a start pose to a goal pose, what Cartesian path between the start and the goal does the end-effector follow? (We assume that execution is successful and the robot never comes close to a singularity.)
    * What is the Configuration Space (C-Space) of a robot?
    * How many dimensions does the C-Space of the Kuka LWR robot (used in the class projects) have?
    * What are different ways in which you might store an obstacle map for a robot?
    * What are the basic building blocks (i.e. external calls) that you need to have available in order to implement a sampling-based motion planning algorithms?
    * If you run the RRT algorithm multiple times on the same problem (same start, goal, and obstacles) will you get the same result? Why?
    * What guarantees does a graph search algorithm such as Dijkstra's or A* offer for a motion planning problem?
    * What guarantees does a sampling-based algorithm such as RRT or PRM offer for a motion planning problem?
    * What is the main advantage of the A* algorithm compared to Dijkstra's algorithm?
    * Is a typical shopping cart (two fixed wheels in the back, two omni-directional casters in the front) a holonomic vehicle? Why?
    * A differential drive robot has positive (forward) velocity at the left wheel, and zero velocity at the right wheel. Assuming no wheel slip, what motion will the robot perform?

### Final Exam Instructions

* About the final exam: 
    *  It will contain 11 questions and students will have 1 hour to complete it.
    * It is open book and open notes. 
    * Students can access the internet during the exam.
    * Calculators are allowed.