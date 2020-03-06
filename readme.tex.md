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
$$T_{ROT}(\theta_{i},z)\cdot T_{TRANS}(d_{i},z)\cdot T_{TRANS}(a_{i},x)\cdot T_{ROT}(\alpha_{i},x)$$

* what this means for the 2-axis planar robot with the DH params we have seen ? we chain the transforms
$$\cos(q_{i})=c_{i}\:\:\:\sin(q_{i})=s_{i}$$
$$T_{ROT}(q_{1},z)\cdot T_{TRANS}(0.5,x)\cdot T_{ROT}(q_{2},z)\cdot T_{TRANS}(0.3,x)=\begin{bmatrix}
c_{1} & -s_{1} & 0 \\ 
s_{1} & c_{1} & 0 \\
0 & 0 & 1\end{bmatrix}\cdot \begin{bmatrix} 
1 & 0 & 0.5\\
0 & 1 & 0 \\
0 & 0 & 1\end{bmatrix}\cdot \begin{bmatrix}
c_{2} & -s_{2} & 0 \\ 
s_{2} & c_{2} & 0 \\
0 & 0 & 1\end{bmatrix}\cdot \begin{bmatrix} 
1 & 0 & 0.3\\
0 & 1 & 0 \\
0 & 0 & 1\end{bmatrix}=\begin{bmatrix}
c_{1} & -s_{1} & 0.5c_{1} \\ 
s_{1} & c_{1} & 0.5s_{1} \\
0 & 0 & 1\end{bmatrix}\cdot \begin{bmatrix}
c_{2} & -s_{2} &  0.3c_{12}+0.5c_{1} \\ 
s_{2} & c_{2} & 0.3s_{12}+0.5s_{1} \\
0 & 0 & 1\end{bmatrix}=\begin{bmatrix}
c_{12} & -s_{12} & 0.3c_{1} \\ 
s_{12} & c_{12} & 0.3s_{1} \\
0 & 0 & 1\end{bmatrix}=^{b}T_{2}$$

* to get to c12 s12 we use trigonometric rules

$$c_{1}c_{2}-s_{1}s_{2}=\cos(\theta_{1}+\theta_{2})=c_{12}$$

* we take a look on our full derived translation matrix and see if it makes sense. if it follows the rules
    * we have a rot matrix with a rotation q1+q2 around z and that makes sense
    * the translation part makes sense trigonometrically according to our sketch

###  3.6 DH Notation Example: SCARA Robot

* 