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
c_{2} & -s_{2} & 0.3c_{2} \\ 
s_{2} & c_{2} & 0.3s_{2} \\
0 & 0 & 1\end{bmatrix}=\begin{bmatrix}
c_{12} & -s_{12} & 0.3c_{12}+0.5c_{1} \\ 
s_{12} & c_{12} & 0.3s_{12}+0.5s_{1} \\
0 & 0 & 1\end{bmatrix}$$

* to get to c12 s12 we use trigonometric rules

$$c_{1}c_{2}-s_{1}s_{2}=\cos(\theta_{1}+\theta_{2})=c_{12}$$

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
$$^{b}T_{4}=\begin{bmatrix} 
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0.5 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix} 
c_{1} & -s_{1} & 0 & 0 \\
s_{1} & c_{1} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix} 
1 & 0 & 0 & 0.7 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix} 
c_{2} & -s_{2} & 0 & 0 \\
s_{2} & c_{2} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix} 
1 & 0 & 0 & 0.7 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix} 
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & q_{3} \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix} 
c_{4} & -s_{4} & 0 & 0 \\
s_{4} & c_{4} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}$$

* when we multiply a trasform matrix containing only translation (R=i) with a transform matrix containing only rotation we can just concatenate the matrices. This works ONLY IN THIS ORDER: Pure transaltion followed by pure rotation
* when we have consecutive Pure Translations we can just add them up
$$^{b}T_{4}=\begin{bmatrix} 
c_{1} & -s_{1} & 0 & 0 \\
s_{1} & c_{1} & 0 & 0 \\
0 & 0 & 1 & 0.5 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix} 
c_{2} & -s_{2} & 0 & 0.7 \\
s_{2} & c_{2} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix} 
c_{4} & -s_{4} & 0 & 0.7 \\
s_{4} & c_{4} & 0 & 0 \\
0 & 0 & 1 & q_{3} \\
0 & 0 & 0 & 1 \end{bmatrix}$$

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
$$q_{i}\overset{Fk}{\rightarrow}^{b}T_{ee}$$

* In real robot applications we care much more about the inverse problem: we know where the target object is (the desired end effector position). this is the relative position to the base frame which we know so we know the Trasform matrix to go from base to end effector. what we dont know are the joint values
$$q_{i}\overset{Ik}{\leftarrow}^{b}T_{ee}$$

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
$$^{b}T_{ee}=\begin{bmatrix}
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
c_{2} & -s_{2} & 0.3c_{2} \\ 
s_{2} & c_{2} & 0.3s_{2} \\
0 & 0 & 1\end{bmatrix}=\begin{bmatrix}
c_{12} & -s_{12} & 0.3c_{12}+0.5c_{1} \\ 
s_{12} & c_{12} & 0.3s_{12}+0.5s_{1} \\
0 & 0 & 1\end{bmatrix}$$

* remember that the following convention is used for trigonometric methods::
$$\cos=c\:\:\:c(q_{1}\pm q_{2})=c_{12}=c_{1}c{2}\mp s_{1}s_{2}$$
$$\sin=s\:\:\:s(q_{1}\pm q_{2})=s_{12}=s_{1}c{2}\pm c_{1}s_{2}$$

* say now we want our end effector at position x=a,y=b we dont care about the orientation of end effector coordinate frame yet just the position. therefore we use only the translation part of the transform matrix frpm base to end effector to derive the joint values (angles θ1 θ2)
$$0.5c_{1}+0.3c_{12}=a\:\:\:0.5s_{1}+0.3s_{12}=b$$

* a good trick to solve the above equation system is to square them both and then add them up
* what we get is 
$$a^{2}+b^{2}=0.25c_{1}^{2}+0.3c_{1}c_{12}+0.09C_{12}^{2}+0.25s_{1}^{2}+0.3s_{1}s_{12}+0.09s_{12}^{2}$$

* we make use of the theorem 
$$\cos(\theta)^2+\sin(\theta)^{2}=1$$

* using this and the main trigornometric methods for θ1+θ2 our equation is simplified
$$a^{2}+b^{2}=0.34+0.3(c_{1}c_{12}+s_{1}s_{12})=0.34+0.3c_{2}$$

* so we have c2.
$$c_{2}=\frac{a^{2}+b^{2}-0.34}{0.3}$$

* the immediate thought it to introduce the solution of c2 into the first equation and solve for c_{1}
* not yet. we have to deal with some cases.also we draw a circle of maximum reach for the robot of radius L1+L2
* CASE 1: if the fraction is > 1 it cannot be a cosine so we have no solutions and theoretically the point is in infinite position outside of max reach
* CASE 2: if the fraction is equal to 1 then c2=1 and we have only 1 solution:
$$q_{2}=0\Rightarrow \left\{\begin{matrix}
0.8c_{1}=a\\ 0.8s_{1}=b
\end{matrix}\right. \Rightarrow q_{1}=\arctan2(b,a)$$

* the solution if c2=1 is that arm is always fully stretched out on the max reach circle
* when we get c1=a we are tempted to do q1=acos(a) but in the range [0,2π] there are muliple angles with c1=a not one. same for sin. there is not a unique solution. combining both equations gives a unique solution using arctan2. arctan2 is an arctangent that looks at the quadrant that the angle that should be in
$$\left\{\begin{matrix}
c_{1}=a\\ s_{1}=b
\end{matrix}\right. \Rightarrow q_{1}=\arctan2(b,a)$$

* so keep in mind. if only cos or sin is given there are multiple solutions. if both are given there is a unique solution for angle
* CASE 3: fraction is between -1 and 1 so -1< c2 <1 then there are 2 possible solutions
$$q_{2}=\left\{\begin{matrix}
\arccos(\frac{a^{2}+b^{2}-0.34}{0.3})\\ 2\pi-\arccos(\frac{a^{2}+b^{2}-0.34}{0.3})
\end{matrix}\right.$$

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
$$^{}bT_{ee}=\begin{bmatrix}
c_{12} & -s_{12} & 0.3c_{12}+0.5c_{1} \\ 
s_{12} & c_{12} & 0.3s_{12}+0.5s_{1} \\
0 & 0 & 1\end{bmatrix}=\begin{bmatrix}
c_{\gamma} & -s_{\gamma} & \alpha \\ 
s_{\gamma} & c_{\gamma} & \beta \\
0 & 0 & 1\end{bmatrix}$$

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
$$^{b}T_{ee}=\begin{bmatrix}
c_{1} & -s_{1} & 0 & 0\\
s_{1} & c_{1} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & -1 & 0 & 0\\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
c_{2} & -s_{2} & 0 & 0\\
s_{2} & c_{2} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 0 & -1 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & q_{3}\\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
c_{3} & -s_{3} & 0 & 0\\
s_{3} & c_{3} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}$$

* we do the matrix multiplications (rotation part gets very complicated)  so we assume we care only about the translation of the end effector and get to 
$$^{b}T_{ee}=\begin{bmatrix}
R & t\\
0 & 1 \end{bmatrix} \:,\:t=\begin{bmatrix}
c_{1}s_{2}q_{3} \\ s_{1}s_{2}q_{3} \\ c_{2}q_{3} \end{bmatrix}$$

* as we said this T matrix is all about the translation of end effector in space. has nothing to do with the rotation of end effector. thats why q4 is missing
* Now we go to Inverse Kinematics taking into consideration only Position and not Orientation. Orientation is too much for doing an analysis on paper. let tools tackle it
* we need to solve the equation system below for q1,q2,q3
$$\left\{\begin{matrix}
x=c_{1}s_{2}q_{3}\\ y=s_{1}s_{2}q_{3} \\ z=c_{2}q_{3}
\end{matrix}\right.$$

* we do the trick of squaring them up and adding them
$$x^{2}+y^{2}+z^{2}=q_{3}^{2}(c_{2}^{2}+s_{2}^{2}(c_{1}^{2}+s_{1}^{2}))=q_{3}^{2}$$
$$q_{3}=\pm\ sqrt{x^{2}+y^{2}+z^{2}}$$

* the negative solution is more for clompleteness. its uncommon to have a prismatic joint extending in reverse direction
* usually in practice we have our system limits. a manufactures always gives joint limits
* so we have 2 SOLUTIONS for q3
* we square x and y and add them to go for q1 q2
$$x^{2}+y^{2}=q_{3}^{2}s_{2}^{2}(c_{2}^{2}+s_{2}^{2})$$

* for s2 we again have 2 solutions
$$s_{2}=\pm \sqrt{\frac{x^{2}+y^{2}}{x^{2}+y^{2}+z^{2}}}$$

* but we have a solutuion for c2 sso we can use atan2
$$c_{2}=\frac{z}{q_{3}} \Rightarrow q_{2}=\arctan2(s2,c2)$$

* as we have 2 solutions for sine we have also 2 SOLUTIONS for q2. also note that we divide by q3. what if q3 is 0. if q3 is 0 x,y,z is 0 so we end up qith a non realistic situation
* now we calculate q1
$$c_{1}=\frac{x}{s_{1}q_{3}}\:\:s_{1}=\frac{y}{s_{2}q_{3}} \Rightarrow q_{1}=\actan2(s_{1},c_{1})$$

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
$$^{b}\bf{T}_{ee}=\left[ \begin{array}{ccc|c} & & & x \\ & \bf{R} & & y \\ & & & z \\\hline 0 & 0 & 0 & 1 \end{array} \right]$$

* We will use the notation from the lectures
$$S_2=\texttt{sin}(q_2)\:\:\:C_2=\texttt{cos}(q_2)$$

* Assume we require the end-effector to be at position  [a,b,c]T , and we do not care about end-effector orientation. Derive the values for the robot joints  q1 ,  q2  and  q3  such that the end-effector achieves the desired position. Be sure to consider all possible solutions.

### 5.1 Differential Kinematics Introduction

* we start again from FW Kinematics on a typical robot arm with 3 links and 3 joints
* FW kinematics is the task of computing the Transform from base to end effector as a functions of the joint angles:
$$^{b}T_{ee}=FWD(q)\:\:where:\:\:q=\begin{bmatrix}q_1 & q_2 & q_3 \end{bmatrix}^{T}$$

* a very common problem when using robots is move the end effector in a specified direction that we know in cartesian space. say from point1 to point2, where we know the change Δx in cartesian space between point 1 and 2. Δx is the difference of 3D point vectors x=[x,y,z]T. if we care also about the orientation apart from position then x=[x,y,z,roll,pitch,yaw] or x=[x,y,z,rx,ry,rz]T. in that case we want to know how much to change the joint values Δq=?
* say we have a welder robot and we need to weld a body so the welding robot has to follow the contour of the surface. we have the path specified in cartesian space. we need to decide how the joints move in order for the robot to follow this path we have specked in cartesian space....
* What the FW Kinematics does for us is that it tells us that the position and orientation of the robot in cartesian space can be expressed as a function of q x=f(q)
* we know that we can easily go from transform matrix to the relative position(and orientation) of end effector in relation to the base frame
$$^{b}T_{ee}=FWD(q)\:\:q=\begin{bmatrix}q1 & q2 & q3\end{bmatrix}^{T}\:\:x=\begin{bmatrix}x & y & z & r_x & r_y & r_z\end{bmatrix}^{T}$$
$$x=f(q)$$

* how to compute Δq. we know that x+Δx=f(q+Δq). we can linearize the function f around the point q
$$x+\Delta x = f(q + \Delta q) = f(q) + \frac{\partial f}{\partial q} \Delta q$$
$$\Delta x = \frac{\partial f}{\partial q} \Delta q\:\:or\:\: \dot{x}=\frac{\partial f}{\partial q} \dot{q}$$

* so our problem is now the parial derivative of the function on the joint values

### 5.2 Manipulator Jacobian

* we need to differentiate the function f but it takes multidimensional input and output
$$q\in \mathbb{R}^{n}\:\:x\in \mathbb{R}^{m}$$
$$n: number\:of\:joints$$
$$m: number\:of\:dimensions$$

* the differentiate functions can be expressed as a m by n matrix of partials
$$\frac{\partial f}{\partial q}=\begin{bmatrix}
\frac{\partial x_1}{\partial q_1} & \frac{\partial x_1}{\partial q_2} & ... & \frac{\partial x_1}{\partial q_n} \\
... & ... & ... & ...  \\
\frac{\partial x_m}{\partial q_1} & \frac{\partial x_n}{\partial q_2} & ... & \frac{\partial x_m}{\partial q_n}
\end{bmatrix}=J$$

* this matrix is called the Jacobian of function f (n columns and m rows)
* the jacobian is the differentiation of function f against q but its valid in aparticular location in input space so the jacobian is a function of q. J(q). as the values of q change so does the matrix
$$\Delta x = J\Delta q\:\:or\:\: \dot{x}=J\dot{q}$$

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
$$^{b}T_{ee}=\begin{bmatrix}
c_{12} & -s_{12} & c_{12}+c_1 \\
s_{12} & c_{12} & s_{12}+s_1 \\
0 & 0 & 1
\end{bmatrix}$$
$$x=\begin{bmatrix}x & y\end{bmatrix}^{T}$$
$$q=\begin{bmatrix}q_1 & q_2\end{bmatrix}^{T}$$
$$x=f(q)$$
$$f=\left\{\begin{matrix}
x=\cos(q_{1}+q_{2})+\cos(q_{1})\\ 
y=\sin(q_{1}+q_{2})+sin(q_{1})
\end{matrix}\right.$$

* we can now calculate the jacobian
$$J=\begin{bmatrix}
\frac{\partial x}{\partial q_{1}} & \frac{\partial x}{\partial q_{2}} \\
\frac{\partial y}{\partial q_{1}} & \frac{\partial y}{\partial q_{2}} 
\end{bmatrix} =\begin{bmatrix}
-\sin(q_{1}+q_{2})-\sin(q_{1}) & -\sin(q_{1}+q_{2}) \\
\cos(q_{1}+q_{2})+\cos(q_{1}) & \cos(q_{1}+q_{2})
\end{bmatrix}$$

* in shortform Jacobian is
$$J=\begin{bmatrix}
-s_{12}-s{1} & -s_{12} \\
c_{12}+c_{1} & c_{12}
\end{bmatrix}$$

* we can now calculate the Jacobian for a particular spot     $q=\begin{bmatrix}\frac{\pi}{4} & -\frac{\pi}{2}\end{bmatrix}^{T}$
* we draw the pose of the robot arm and calculate the Jacobian
$$J(\frac{\pi}{4},\frac{\pi}{2})=\begin{bmatrix}
-s(-\frac{\pi}{4})-s(\frac{\pi}{4}) & -s(-\frac{\pi}{4}) \\
c(-\frac{\pi}{4})+c(\frac{\pi}{4}) & c(-\frac{\pi}{4})
\end{bmatrix}$$=\begin{bmatrix}
0 & \frac{\sqrt{2}}{2} \\
\sqrt{2} & \frac{\sqrt{2}}{2}
\end{bmatrix}=\frac{\sqrt{2}}{2}\begin{bmatrix}
0 & 1 \\
2 & 1
\end{bmatrix}

* so we have the jacobian for this position... we can use the Δ equation and given the Δx calc the Δq needed $\Delta q= J^{-1}\cdot \Delta x$
* the jacobian inverse for this position is
$$J^{-1}=\frac{\sqrt{2}}{2}\begin{bmatrix}
-1 & 1 \\
2 & 0
\end{bmatrix}\:\:\:\Delta x =\begin{bmatrix} 1 \\ 0 \end{bmatrix} \Rightarrow \Delta q = \frac{\sqrt{2}}{2}\begin{bmatrix}
-1  \\ 2 
\end{bmatrix}\$$

* it makes sense q1 became smaller and q2 larger. so its correct
* be careful with signs
* another example is if we want the end effector to move up
$$\Delta x =\begin{bmatrix} 0 \\ 1 \end{bmatrix} \Rightarrow \Delta q = \frac{\sqrt{2}}{2}\begin{bmatrix}
1  \\ 0 
\end{bmatrix}\$$

* this also makes sense. only q1 becames larger
* remember that we need to recompute jacobian for every position

### 5.4 Singularities

* we will try to calculate the Jacobian for the previous example when q2 = π.  it will be
$$J=\begin{bmatrix}
0 & s_1 \\ 0 & -c_1
\end{bmatrix}$$

* this is an important case. robot arm has fully folded onto itself
* if i multiply the Jacobian with Δq=[Δq1,Δq2]T it doesnt matter what i change in q1. it wont have any effect in positon as 1st column of Jacobian is 0
* in the sketch we can verify that. if the robot rotates around q1 end effector is in 0,0 position
* in this situation q1 lost its ability to move the end effector
* this is a problem. another problem is that the determinant of the Jacobian is 0. we cannot invert it and we cannot compute joint val move if position changes (but it cant)
* if we try to compute Δq for an arbitrary Δx. the equation system that we have (see below) is unsolvable
$$\left\{\begin{matrix}
s_{1}\Delta q_{2}=\Delta x\\ 
-c_{1}\Delta q_{2}=\Delta y
\end{matrix}\right. \Rightarrow \frac{\Delta x}{\Delta y} = -\frac{s_1}{c_1}$$

* movement is possible only if the derived equation holds. if not we cannot satisfy the equation
* for the specific robot config in this position the only movement possible is along the tangent of the circle arount the second joint if q2 changes.
* so we are locked. what happens if i am close to being locked. if q2 is not π but very close to it
* in that case in the jacobian instead of 0 we would have 2 very small vals ε1 and ε2. also the determinant of the Jacobian will be non-zero
* then we can attempt to solve the equation system and calculate the Jacobian inverse then calculate the Δq for a small Δx in this position
$$\Delta q_{2}=\frac{\Delta x -\frac{\varepsilon_{1}}{\varepsilon_{2}}\Delta y}{\frac{\varepsilon_{1}}{\varepsilon_{2}}c_{1}+s_{1}}$$
$$\Delta q_{2}=\frac{\Delta y+c_{1}\Deltaq_{2}}{\varepsilon_{2}}$$

* remember that ε is very small. what if we write a piece of SW that takes in Δx calculates jacobian inverse and Δq and sents it to the robot given that the robot is close to the q2=π position
* as we divide for ε Δq1 is huge and Δq is analogus to q dot aka speed. so robot will attempt to cover instanlty a huge distance
* this will destroy the robot!!!!!!!!!!
* so being close to a border position commanding the robot can cause instability
* this kind of position is called a Singularity. these positions occur when the determinant of the Jacobian is 0
$$Singularity\:\left | J \right |=0$$ 
* Being in a singularity means that a joint has lost its ability to move the robot
* Also being in a singularity measn we are constrained to move only in a specific direction only
* Being IN a singularity (locked) is better than being very close to a singularity. then asking for a finite movemtn can result in an infinite movement in joint space
* In a robot control software we must avoid singularities and approaching them
* this is done using a SW library that calculates the matrix condition (determinant). if its good then we are safe.
* if q2=0 (arm fully extent the jacobian is
$$J=\begin{bmatrix}
-2s_1 & -s_1 \\
2c_1 & c1
\end{bmatrix}$$ 

* what we see is that columns are not lineraly indipendent. the determinant is 0. the robot is fully extent.
* if we move q1 the robot will move arount the max circle tangent at that position. same for q2. 
* the only possible movement is along the tangent line regardless of the joint angle changing val
* again we have the instability problem. so we lost the ability to move except on one line

### 5.5 Differential Kinematics Example- Spherical Robot

* we look at a 3D example the spherical robot.
* the position of the end effector in space is $x=\begin{bmatrix}x y z\end{bmatrix}^{T}$ as we dont care about the orientation
* the spherical robo has 3 DOF (3 Joints) so the q vecor is $\begin{bmatrix}q_{1} q_{2} q_{3}\end{bmatrix}^{T}$
* we recall the relationship of endeffector position with joint values
$$\begin{bmatrix} x \\ y \\ z \end{bmatrix}=\begin{bmatrix}
c_{1}s_{2}q_{3} \\ s_{1}s_{2}q_{3} \\ c_{2}q_{3} \end{bmatrix}$$

* the general formula for the Jacobian is:
$$J=\begin{bmatrix}
\frac{\partial x}{\partial q_{1}} & \frac{\partial x}{\partial q_{2}} &  \frac{\partial x}{\partial q_{3}} \\
\frac{\partial y}{\partial q_{1}} & \frac{\partial y}{\partial q_{2}} &  \frac{\partial y}{\partial q_{3}} \\
\frac{\partial z}{\partial q_{1}} & \frac{\partial z}{\partial q_{2}} &  \frac{\partial z}{\partial q_{3}}
\end{bmatrix}$$
* we compute the Jacobian for the general x case
$$J=\begin{bmatrix}
-s_{1}s_{2}q_{3} $ c_{1}c_{2}q_{3} $ c_{1}s_{2} \\ 
c_{1}s_{2}q_{3} & s_{1}c_{2}q_{3} & s_{1}s_{2} \\ 
0 & -s_{2}q_{3} & c_{2} \end{bmatrix}$$

* we will now calculate the determinant of the Jacobian to be able to avoid the Singularities
$$\left | J \right |=-s_{2}q_{3}^{2}$$

* we solve for 0 to detect the singular positions to avoid $\left | J \right |=0 \:\Rightarrow\:s_{2}=0\:\Rightarrow\:q_{2}=0,\pi\:\:q_{3}=0$

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
$$q=\begin{bmatrix}q_1 & q_2 & q_3 \end{bmatrix}^{t}\in \mathbb{R}^{n}$$

* In Cartesian Space
    * we talk about relative position and orientation  of ee regarding base frame (translation and rotation along each axis)
        * so we work in 6dimensional space if we care about position and orientation. if we care only about position in 3d space we work in 3d. if we cae about position&orientation in 2D plane we work in 3D
        * we call this space task space because it represents target position where the endeffector has to go
$$q=\begin{bmatrix}x & y & z & r_x & r_y & r_z\end{bmatrix}^{t}\in \mathbb{R}^{6}$$

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
$$x=f(q)\:\Rightarrow\:J=\begin{bmatrix}
\frac{\partial x_1}{\partialq_1} & ... & \frac{\partial x_1}{\partialq_n} \\
... & ... & ... \\
\frac{\partial x_m}{\partialq_1} & ... & \frac{\partial x_m}{\partialq_n}
\end{bmatrix}$$

* also:
$$J\Delta q=\Delta x$$
$$J\dot{q}=\dot{x}$$

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
$$^{b}T_{ee}=\begin{bmatrix}
c_{1} & -s_{1} & 0 & 0\\
s_{1} & c_{1} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & e_1\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 0 & -1 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
c_{2} & -s_{2} & 0 & 0\\
s_{2} & c_{2} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 0 & -1 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & q_3\\
0 & 0 & 0 & 1 \end{bmatrix}$$

* remember teh multiplication and trick (trnaslate+rotation) and get the final transform matrix
* also if we say we care only for the translation part of forward kinematics our equations are simplified. we keep only the transaltion part of last matrix replacing it with a [0 0 3 1]T vector
$$^{b}T{ee}=\begin{bmatrix}
c_1(s_2q_3+l_1) \\
s_1(s_2q_3+l_1) \\
-c_2q_3 \\
1
\end{bmatrix}$$

* we do a sanity check projecting the end effector point from the sketch along the 3 axis
* we care only about position so x will be x=[x y z]T
* we calculate the Jacobian
$$J=\begin{bmatrix}
\frac{\partial x}{\partial q_{1}} & \frac{\partial x}{\partial q_{2}} &  \frac{\partial x}{\partial q_{3}} \\
\frac{\partial y}{\partial q_{1}} & \frac{\partial y}{\partial q_{2}} &  \frac{\partial y}{\partial q_{3}} \\
\frac{\partial z}{\partial q_{1}} & \frac{\partial z}{\partial q_{2}} &  \frac{\partial z}{\partial q_{3}}
\end{bmatrix}=\begin{bmatrix}
-s_1(s_2q_3+l_1) & c_1c_2q_3 & c_1s_2 \\
c_1(s_2q_3+l_1) & s_1c_2q_3 & s_1s_2 \\
0 & s_2q_3 & -c_2 \end{bmatrix}$$

* we calculate its determinant (choose the easiest row to expand, add signs to heve the sign, multiply) remember s2+c2=1
$$\left | J \right |=+s_2q_3[-s_1^2s_2(s_2q_3+l_1)-c_1^2s_2(s_2q_3+l_1)]+c_2[-s_1^2c_2q_3(s_2q_3+l_1)-c_1^2c_2q_3(s_2q_3+l_1)]=q_3s_2^2(s_2q_3+l_1)+q_3c_2^2(s_2q_3+l_1)=q_3(s_2q_3+l_1)$$

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
$$^{b}T_{ee}=\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & q_1\\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
c_{2} & -s_{2} & 0 & 0\\
s_{2} & c_{2} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & -1 & 0 & 0\\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & q_3\\
0 & 0 & 0 & 1 \end{bmatrix}=\begin{bmatrix}
c_{2} & -s_{2} & 0 & 0\\
s_{2} & c_{2} & 0 & 0 \\
0 & 0 & 1 & q_1 \\
0 & 0 & 0 & 1 \end{bmatrix}\cdot \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 0 & 1 & q_3\\
0 & -1 & 0 & 0\\
0 & 0 & 0 & 1 \end{bmatrix}=\begin{bmatrix}
c_{2} & 0 & -s_{2} & -s_{2}q_3\\
s_{2} & 0 & c_{2} & c_{2}q_3 \\
0 & -1 & 0 & q_1 \\
0 & 0 & 0 & 1 \end{bmatrix}$$

* we do the IK analysis for position only assuming we want our end effector to end up in position [a,b,c]T
* the translation part of the Transform matrix has to put the end effector in the position we want so we have our equation system
$$\left\{\begin{matrix}
-s_{2}q_3=a \\ c_{2}q_3=b \\ q_1=c \end{matrix}\right.$$

* c we have. its q1 for the other two we square them up and add them το get $q_3=\pm \sqrt{a^2+b^2}$
* we start investigating solutions:
* if $a^2+b^2=0$ we have one solution for q3=0
* if $a^2+b^2\neq 0$ we have 2 solutions for q3 so we use atan2 so for q2
$$s_2=-\frac{a}{q_3}\:\:c_2=\frac{b}{q_3}\:\Rightarrow\:q_2=\arctan2(s_2,c_2)$$

* we do differential kinematics analysis calculating the Jacobian
$$J=\begin{bmatrix}
0 & -c_2q_3 & -s_2 \\
0 & -s_2q_3 & c_2 \\
1 & 0 & 0 \end{bmatrix}$$

* the determinant of the Jacobian is
$$\left | J \right |=-c_2^2q_3-s_2^2q_3=-q_3$$

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

* Compute the translation part of the Forward Kinematics transform $^{b}\bf{T}_{ee}$ from the base of the robot to the end-effector. In other words, derive the expressions for the components of the  $3 \times 1$ vector  $\bf{t}$  below:
$$^{b}\bf{T}_{ee}=\left[ \begin{array}{ccc|c} & & &  \\ & \bf{R} & & \bf{t} \\ & & &  \\\hline 0 & 0 & 0 & 1 \end{array} \right]$$

* We will use the notation from the lectures, i.e. 
$$S_2=\texttt{sin}(q_2)$$
$$S_{23}=\texttt{sin}(q_2+q_3)$$

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
* If we get Δx by the user we have to convert it to velocity in the direction of the Δx $\dot{x}=\rho \Delta x$ where ρ is the proportional gain. the equation is a proportional controller. from this cartesian velocity we will compute the velocity to send to the joints using the differential analysis equation and the jacobian $J\dot{q}=\dot{x}$ this q dot is what we send to the robot
* this works in posiiton and orientation
* in our case where the Jacobian comes from. in our lectures we started with forward kinematic analysis. we had $x=f(q)$ as result of FWD Kinematics. If we have the analytical function of FW kinematics we just have to calculate partial derivatives and build the Jacobian matrix $J=\frac{\partial f}{\partial q}$
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
$$^{A}V_{A} \in \mathbb{R}^6 = \begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{z} \\ \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}$$

* ω is the angular speed of joints
* we want to know what is the resulting velocity of B in coordinate frame B: $^{B}V_{B}=?$
* We know the transform from A to B is:
$$^{A}T_{B}=\begin{bmatrix} ^{A}R_B & ^{A}t_B \\ 0 & 1\end{bmatrix}$$

* The velocity of B is a 6by6 matrix:
$$^{B}V_{B}=\begin{bmatrix}
^{B}R_{A} & -^{B}R_{A}\cdot S(^{A}T_{B}) \\
0 & ^{B}R_A \end{bmatrix}\cdot ^{A}V_A$$

* what we understand is that the angular velociity of B will be the angular velocity of A rotated by $^{B}R_A$ which is the transpose of $^{A}R_B$
* posiional (translation) part of velocities plays no part in angular velocity
* the translation part of the velocity conversion has a rotation part of the translational velocity
* also the translational velocity has to do with the rigid body which is as well rotated..
* matrix S is a skeyw-symmetrix matrix: 
$$S(\begin{bmatrix}x & y & z\end{bmatrix})=\begin{bmatrix}
0 & -z & y \\
z & 0 & -x \\
-y & x 0 \end{bmatrix}$$

* S matrix has the nice property of $S(a)\cdot b = a \times b$ so an easy way to express a cross product as matrix multiplication
* so  the upper right element is the cross product of the model arm with the rotation of frame from joint to end effector which sets the translational velocity conversion with the pure rotation
* the trransform matrix producing the rotation matrix and translation matrix we use for the velocity conversion we get from FWD Kinematic analysis
* for conversion from joint j to end effector ee
$$V_{j}=\begin{bmatrix}
^{ee}R_{j} & -^{ee}R_{j}\cdot S(^{j}T_{ee}) \\
0 & ^{ee}R_j \end{bmatrix}$$
$$v_{ee}=V_{j}\cdot v_{j}$$

* the velocity of joint j can be of anytype. but assuming its a revolute joint we say th ony possible velocity is a rotation around z so
$$v_j=|begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ \dot{q}_{j}\end{bmatrix}$$

* in this case only the last 6th column of Vj matters [:,5]in calculating ee velocity which will be 6by1 vector
* robot have many joints. what if multiple joints move simultaneously. assuming all arrre revolute around z::
$$v_{ee}=V_{0}[:,5]\cdot \dot{q_0} + V_{1}[:,5]\cdot \dot{q_1} + ... + V_{n-1}[:,5]\cdot \dot{q_{n-1}} $$
$$v_{ee}=\sum_{i=0}^{n-1}V_i[:,5]q_i$$ 

* this can be presented in matrix multiplication form wher V[:,5] is a 1xn vector multiplined with dotq a nx1 vector 
* we express the relation witht the jacobian notion where
$$V_{ee}=\begin{bmatrix} V_0[:,5] & V_1[:,5] & ... & V_{n-1}[:,5]\end{bmatrix}\cdot \begin{bmatrix} 0 \\ ... \\ 0 \\ \dot{q_{n-1}} \end{bmatrix}$$
$$V_{ee}=J\cdot \dot{q}$$

* this is the Numerical Jacobian. 
* Knowing only the relative transforms from joint coordinate frames to end-effector coordinate frame we build the transmission matrices V
* Vee is expressed in end effector coordinate frame
* Δx we were given is  in relation to base coordinate frame
* if we have the $^{b}T_{ee}=\begin{bmatrix} ^{b}R_{ee} & ^{b}t_{ee}\\ 0 & 1\end{bmatrix}$
* then what we have as Velocity of the end effector exrpessed in its own coordinate frame related to the speed of x relted to the base frame $\dot{x}=\rho \cdot \Delta x$is
$$V_{ee}=\begin{bmatrix} ^{ee}R_{b} & 0\\ 0 & ^{ee}R_{b}\end{bmatrix}\cdot \dot{x}$$

* where the see that the speed on the base frame isthe end effecrtor frame velocity rotated
* so the way to work is we are given Δx => xdot => Vee => calculate the Jacobian => get qdot

### 7.3 Singularity Avoidance: Jacobian Pseudoinverse

* we recap:
$$J\cdot \dot{q}=V_ee$$
$$J \in \mathbb{R}^{m\times n}$$

* n is the number of robot joints and m is the number of variables we are controlling in end effector
* we compute qdot
$$\dot{q}=J^{-1}V_{ee}$$

* To have an inverse of a Jacobian matrix it must have equal dimensions m=n and Jacobian must be a full rank
$$J\dot{q}=JJ^{-1}V_ee$$ 

* the above shows that J-1 must be at least the Right side Inverse of J so the constraint is m<=n but Jacobian still has to be full rank
* The problem is that as Jacobian approaches the singularity $\dot{q} \rightarrow \infty$ so again we send infinite velocities to the robot
* We will use linear algrbra and the Singular Value Decomposition of a Matrix
* we write the jacobian as the product of 3 matrices $J=U\SigmaV^T$ where J is m x n
* $U \in \mathbb{R}^{m \times m}\:\: UU^T=i$ so U is square and orthogonal
* $\Sigma \in \mathbb{R}^{m \times n}$ Σ is diagonal matrix
* $V\in \mathbb{R}^{n \times n}\:\: VV^T=i$ so V is square and orthogonal
* if m<=n:
$$\Sigma = \begin{bmatrix}
\sigma_1 $ ... $ 0 $ 0 \\
... & ... & ... & 0 \\
0 & ... & \sigma_m & 0 \end{bmatrix}$$

* the sigma values on the diagonal ara on descending order so σ1 >= σ2 >= ... >= σm >= 0
* also if n = RANK(J) all values past it will be 0 $\sigma_i = 0 \forall i \geq n $
* if the Jacobian is Rank defective so its rank is m-1 σm = 0. 
* this is a way to tell by eye the rank of a matrix 
* a robust way to tell numerically that a matrix is approaching the singularity, being close to lose rank iswhen: $\frac{\sigma_m}{\sigma_1} = \epsilon \rightarrow 0 $
* the program might decide when seeing is close to lose rank aka approaching singularity to stop moving to avoid issuing infinite command for protection. this is not optimal as the robot is stuck.
* the correct apporach is to allow to go back but not towards the singularity
* we will see another better way..
* we ll see the matrix we get when we invert Σ
$$\Sigma^{-1}=\begin{bmatrix}
\frac{1}{\sigma_1} & ... & 0 \\
... & ... & ... \\
0 & ... & \frac{1}{\sigma_m} \\
0 & ... & 0 \end{bmatrix}$$
$$\Sigma\cdot Sigma^{-1}=i$$
$$V\cdot\Sigma^{-1}U^{T}=J^{-1}$$

* the last equation is fine when Jacobian holds rank. as it starts to lose rank the 1/σm gets bigger so the inverse Jacobian gets bigger. when σm is 0 we cannot even invert the Jacobian as we get infinity
* the cheap trick is when we see that $\frac{\sigma_m}{\sigma_1}<\epsilon\$ then in the position of $\frac{1}{\sigma_m}$ we put 0 in the Σ-1. all the οhter diagonal vals we invert normally and continue computation. but values that go to infinity we replace them with 0
* then we call the matrix
$$\Sigma^{+}$$
$$V\cdot\Sigma^{+}U^{T}=J^{+}$$

* J+ is a very importan matrix. its called the Jacobian pseudo inverse. it has some very important properties for us
* If J is a full rank Jacobian $JJ^{+}=i$
* If J is a low rank Jacobian $JJ^{+} \neq i$
* If we compute qdot with J+ $\dot{q}=J^{+}V_ee$ we get some excellent properties:
    * if J is full rank it is an exact solution of $$\dot{q}=V_ee$
    * if J is low rank the angular velocities computed wont allow any additional movement towards the singularity but will allow any movement that does not get us closer to the singularity. SWEETT!! we get the best of both worlds
* In Practice any linear algebra lib has methods to compute the pseudo inverse `numpy.linalg.pinv(J,epsilon)`

### 7.4 Putting It All Together: Cartesian Control

* How we do Cartesian Control?
**Given:**
* the coordinate frames for all the joints: $\left \{  j \right \}$
* the transform matrix from the base to the current position of the end effector $^{b}T_{ee_current}$
* the transform matrix from the base to the desired position of the end effector $^{b}T_{ee_desired}$
**Assume:**
* all joints are revolute
**Output**
* Joint angle velocity $\dot{q}$
**Steps**
* from the 2 base to end-effector Transform matrices we compute a Δx: $^{b}T_{ee_cur},^{b}T_{ee_des}\rightarrow \Delta x$
* from the Δx we get xdot multiplying with the gain, the gain is set by trial and error (proportional control) $\dot{x}=\rho \Delta x$
* using the transform matrix we transform xdot to desired velocity of end effector in its own coordinate frame  $^{b}T_{ee}, \dot{x} \rightarrow V_{ee}$
* for each joint j we compute matrix Vj relating velocity of joint at the joint coordinate frame to velocity of the ee expressed in its own coordinate frame. we keep only the last coolumn of this matrix (as we care for angular velocity in z) $FOR\:EACH\:JOINT\:j \rightarrow V_j\:,\:V_j[:,5]\cdot \dot{q_j}=V_{ee}$
* We assemble these columns in block column form and get the jacobian $ASSEMBLE\:V_j[:,5]\:INTO\:J\:,\:J\dot{q}=V_{ee}$
* we  compute the pseudoinverse of J $J^{+}$ with ε
* we compute $\dot{q}=J^{+}V_{ee}$
* we send $\dot{q}$ to the robot
* In practice we also put some safeguards along the computations. in robotics.. if something goes wrong the robot might cause an accident
**Safeguards**
* Scale $\dot{x}$ such that $\left \| \dot{x} \right \| \leg t_1$  less than a threshold, to prevent sending too much movement to the robot
* Scale $\dot{q}$ such that $\left \| \dot{q} \right \| \leg t_2$  less than a threshold, to prevent sending too much movement to the robot
* Alternatively  scale $\dot{q}$ such that $ \dot{q_i}  \leg t_3 \forall i$  less than a threshold, to prevent sending too much movement to the robot

### 7.5 Redundant Robots: Null Space Control

* Redundant robots are used mostly in research. Is one that has more than the smallest number of DOF needed to achieve any combination of position and orientation for end effector
* In 3D with position and orientation this means >6 DOF. it will give me more ways to achieve the same orientation and position (infinite in some configs)
* we write our main equation $J\dot{q}=V_ee$
* we want to know if thre are angular velocities different than 0 so that when multiplied with J they are 0 $\exists \dot{q_n} \neq 0\:such\:that\:J\dot{q_n}=0$
* what we are really asking is if there are joint velocities that produce no velocity to the end effector $J\dot{q_n}=0$ means that dotqn is in the null space of the Jacobian
* if m >= n the above happens only at singularities
* what is the dimensionality of the null space of the Jacobian: at least 1
* if m < n (if we have more joints than we need) the lin algebra rank-nullity theorem tells us that thats always the case. it means that at any moment we can move the joints in a way that does not produce movement to end effector
* the way to compute the qdot in the null space of the Jacobian is by projecting the  input into the null space. 
* to do it if we have any joint velocity q dot we left multiply it with 1-J+J then its guaranteed to always be in the null space of the Jacobian
$$\dot{q_n}=(i-J^{+}J)\dot{q} \Rightarrow J\dot{q_n}=(J-JJ^{+}J)\dot{q}$$

* this is because $JJ^{+}J=J$ which holds always: whether J is full rank or not. 
* if J is full column rank: $JJ^{+}=i$. 
* if J is full row rank: $J^{+}J=i$
* if i  calculate the Jacobian pseudoinverse and the the solution for dotq $\dot{q_s}=J^{+}V_ee$
* before sending it to robot we can have another goal for the end effector. to move the joints without moving the end effector 
* then what I will send to the robot will be $\dot{q_S}=J^{+}V_ee+\dot{q_n}
* in that case we dont mess up with out primary goal but get the second goal by choosing $\dot{q}$ so that i can get the $\dot{q_n}$ 
* we can use it to avoid obstacles
* In Cartesian control 
    * we send commands in Cartesian Space of the end effector using the Jacobian
    * we defend against singularities
* Cartesian control is all about moving the end effector between two points by the shortest possible path

### Project 4

**Project 4 Description**

* In this project you will apply what you have learned about differential kinematics, the numerical computation of the Jacobian and singularity avoidance. You will write a Cartesian controller for the same 7-joint robot arm that you already know from the last project. This controller will allow you to interactively move the end-effector by dragging around an interactive marker.
* As the robot has 7 joints, it has a redundancy and you can also implement the null-space control. In this assignment, the goal of null-space control is to change the value of the first joint (thus turning the "elbow" of the robot) without affecting the pose of the end-effector (specified above as a primary goal). However, null-space control implementation is optional and will not be part of the grade. We still encourage you to test your understanding of the concepts by implementing it, without having to hit specific targets for the grade.

*The cartesian_control(...) function*

* You are given starter code that includes a `cartesian_control` package, which in turn contains the `cartesian_control.py` file you must edit. Specifically you must complete the `cartesian_control` function. The arguments to this function are all the parameters you need to implement a Cartesian controller:
    * `joint_transforms`: a list containing the transforms of all the joints with respect to the base frame. In other words, `joint_transforms[i]` will contain the transform from the base coordinate frame to the coordinate frame of joint `i`.
    * `b_T_ee_current`: current transform from the base frame to the end-effector
    * `b_T_ee_desired`: desired transform from the base frame to the end-effector
* In addition, the parameters below are relevant only if you are also choosing to implement null-space control:
    * `red_control`: boolean telling you when the Cartesian Controller should take into account the secondary objective for null-space control. This is only set to True when the user interacts with the control marker dedicated to the first joint of the robot.
    * `q_current`: list of all the current joint positions
    * `q0_desired`: desired position of the first joint to be used as the secondary objective for null-space control. Again, the goal of the secondary, null-space controller is to make the value of the first joint be as close as possible to `q0_desired`, while not affecting the pose of the end-effector.
* The function must return a set of joint velocities such that the end-effector moves towards the desired pose. If you are also implementing null-space control and `red_control` is set to true, the joint velocities must also attempt to bring `q[0]` as close to `q0_desired` as possible, without affecting the primary goal above.

**Algorithm Overview**

* The problem we are aiming to solve with a Cartesian controller is the following: If we have a robot that allows us to directly set joint velocities, what velocities do we set such that the robot achieves a desired end-effector position? The algorithm follows the steps presented in detail in the lectures and recapped in the lecture video "Putting It All Together: Cartesian Control".  Given the parameters listed above, a high level overview of the algorithm to achieve this could be:
    * compute the desired change in end-effector pose from `b_T_ee_current` to `b_T_ee_desired`. A couple of important points: a) at some point, you will need to go from a desired rotation expressed as a rotation matrix to the same desired rotation expressed as 3 numbers (rotations around the x, y and z axes). In the same file, you will find a helper function called rotation_from_matrix() that, given a rotation expressed as a matrix, gives you the same rotation expressed as an angle around an axis in space. You might find this function helpful. b) it is also important to remember that, eventually, you will need to compute the desired pose change of the end-effector expressed in its own coordinate frame, not in the base frame.
    * convert the desired change into a desired end-effector velocity. This is essentially a velocity controller in end-effector space. The simplest form could be a proportional controller, where the velocity is equal to the desired change scaled by a constant. You also might want to normalize the desired change if it is larger than a certain threshold to obtain a maximum end-effector velocity (e.g. 0.1 m/s and 1 rad/s) 
    * Numerically compute the robot Jacobian. For each joint compute the matrix that relates the velocity of that joint to the velocity of the end-effector in its own coordinate frame. Assemble the last column of all these matrices to construct the Jacobian
    * Compute the pseudo-inverse of the Jacobian. Make sure to avoid numerical issues that can arise from small singular values
    * Use the pseudo-inverse of the Jacobian to map from end-effector velocity to joint velocities. You might want to scale these joint velocities such that their norm (or their largest element) is lower than a certain threshold
    * If you are not implementing null-space control, you can return these joint velocities.
    * If you choose to implement null-space control of the first joint, find a joint velocity that also brings that joint closer to the secondary objective. Use the Jacobian and its pseudo-inverse to project this velocity into the Jacobian nullspace. Be careful to use the 'exact' version of the Jacobian pseudo-inverse, not its 'safe' version. Then add the result to the joint velocities obtained for the primary objective
    * Return the resulting joint velocities, which will then be sent to the robot    

**Setup**

* As in the previous project, please make sure that the first thing you do in your workspace is to `source` the `setup_project4.sh` script. This starts up the simulated robot and the interactive controls you can use to move it. You can now press the 'Connect' button and you should see both the robot and the controls.

![image](http://roam.me.columbia.edu/files/seasroamlab/imagecache/103X_P4_1.png)

**Grading/ Debug**

* After you have completed the Cartesian controller, you can run your code using `rosrun cartesian_control cartesian_control.py` and use the interactive controls to move the robot around. What you should see is that the robot moves such that the end-effector follows where you drag your mouse.  The behavior should resemble the demo shown in the lecture videos. In particular, when you stretch the robot all the way out (get close to a singularity) there should be no sudden jerks or jumps.
* For this project, we will also give you a piece of grader code that you can use to test your controller. You can right-click on the interactive control and click 'run grader'. This will set goals for the robot and wait for it to reach them. The first goal will involve a pure translation, the second a pure rotation and the final a combination of translation and rotation. The grader will wait for the robot to reach each goal for 10 seconds. If the robot does not reach a goal in time it will time out and the grader will continue with the next goal. If your robot moves towards the goal but does not reach it in time, consider increasing the gain on your velocity controller or increasing the safety thresholds on end-effector and joint velocities. 

**How do run this project in my own Ubuntu machine?**

* Launch Project 4, then in Vocareum click Actions>Download Starter code. This will download all the files you need to make the project run locally in your computer.
* Install the needed ROS package(s). Run the following lines on your terminal:
```
sudo apt-get update
sudo apt-get install ros-kinetic-urdfdom-py
```

* Replace kinetic with the ROS version that you are running on your local machine.
* IGNORE all the files other than `catkin_ws`, `lwr_defs` folders, and `kuka_lwr_arm.urdf`. Put the `catkin_ws` and the `kuka_lwr_arm.urdf` file in your home directory. Now grab the `lwr_defs` folder and move it inside your `catkin_ws/src` folder with the rest of the packages.
* The downloaded files are structured as a catkin workspace. You can either use this structure directly (as downloaded) and build the workspace using the `catkin_make` command or use whatever catkin workspace you already had, and just copy the packages inside your own src folder and run the catkin_make command. If you are having troubles with this, you should review the first ROS tutorial "Installing and configuring your ROS Environment".
* Once you have a catkin workspace with the packages inside the src folder, you are ready to work on your project without having to make any changes in any of the files. Navigate to the catkin workspace folder and build the workspace using the command `catkin_make`.
* OTE: You can source both your ROS distribution and your catkin workspace automatically everytime you open up a terminal automatically by editing the `~/.bashrc` file in your home directory. For example if your ROS distribution is Kinetic, and your catkin workspace is called `project4_ws` (and is located in your home directory) then you can add the following at the end of your ``.bashrc` file:
```
source /opt/ros/kinetic/setup.bash
echo "ROS Kinetic was sourced"
source ~/project4_ws/devel/setup.bash
echo "project4_ws workspace was sourced"
```

* This way every time you open up a terminal, you will already have your workspace sourced, such that ROS will have knowledge of the packages there. 
* Before moving forward, if you haven't followed the instructions of NOTE, you will need to source ROS and the catkin workspace every time you open a new terminal. To run the project, first open up a terminal and type `roscore`. In the second terminal (remember to source ROS and the catkin workspace if you didn't do NOTE) make sure you are in your home directory and run `rosparam set robot_description --textfile kuka_lwr_arm.urdf`, followed by `rosrun robot_sim robot_sim_bringup`.
* On another 2 separate terminals you need to run the scripts for the robot state publisher and interactive markers:  `rosrun robot_state_publisher robot_state_publisher` and `rosrun cartesian_control marker_control.py`. Note that you can find these lines from `setup_project4.sh` in the starter code. Finally you can run your own script in another terminal: `rosrun cartesian_control cartesian_control.py`
* NOTE: you can safely ignore the messages "multiple times: collisionScalar element defined multiple times" on the console.
* Now we can open up Rviz using `rosrun rviz rviz`. Inside Rviz, first change the Fixed Frame to "world_link". Then click Add and select RobotModel from the list of options. At this point you should see the robot arm standing straight up. To add the interactive marker needed to command the robot around, click Add and select "InteractiveMarkers" from the list. Then on the left navigation plane, expand the InteractiveMarkers object, and click on Update Topic > /control_markers/update.
* Now you should see the robot in Rviz, along with an interactive marker to command different positions for the end effect. Once your code works, the robot will follow whatever command is issued by moving this marker around. 

**How to compute delta_X with helper function for the rotation part.**

* Remember you need to do this in the End Effector frame. Hence to compute *delta_X:*
* Using the knowledge of the transformations *b_T_ee_current* and *b_T_ee_desired* you can compute the transform that goes from ee_current to ee_desired. This transformation contains the translation part and rotation part that you need in the proper frame.
* The translation part is right there in the transformation matrix, no need for further processing, just extract it.
* However, the rotation part is a rotation matrix, which you will have to work to transform into a 3 number delta vector to represent the change in pose: *delta_w = [a,b,c]^t* and be able to build your *V_ee* vector. This is where the helper function comes in and gives you the rotation in the angle-axis representation (which is basically an angular velocity vector, i.e, what you were looking for).
* At this point you have your delta translational and rotational. We recommend to work the translation and rotation parts independently:
* You compute your EE translational delta, multiply by a gain to obtain your EE translational velocity, and then scale the whole translational vector just in case its norm is too big (that way you cap the maximum velocity). You do the same for rotation, and then at the end you just put the two together to build your *V_ee*.

## Week 8: Motion Planning I

### 8.1 Robot configuration space (C-space)

* Motion Planning is the problem of computing a path for a robot to go from point A to point B
* The caresian control we have seen is a form of path planning. we are computing a path for the robot to get end effector from point A to point B. to path its such so that end efector goes from a to b in a straight line aka the shortest distance. actually the path we compute is joint movement
* what if the straight line to the goal is blocked by an obstacle. how do we go from point A topoint B
* sometimes the robot cannot afford to go to the goal by the shortest path. it has to go around obstacles. it has to plan a path
* in motion planning we assume that we have a precomputed robot configuration that places the end effector to the right (desired) location using inverse kinematics
* the problem is how to we plan a path to go from A to B without hitting the obstacles
* this problem is not only applicable to robot arms but to mobile robots as well
* in mobile robots the whole robot moves
* in robot arms the base is fixed only the joints move
* in order to see that they are similar we have to think for the robot arm not in cartesian space but in joint space (configuration space)
* we go again to 2D drawing a 2-link planar robot arm with q1 and q2
* in cartesian space we have **x**=[x,y]T as a point desc
* a point in joint space  will be **q**=[q1,q2]T
* both spaces are 2D
* we do a plot in configuration space q1,q2. 
* in which cases we have robot configurations that bring collision with an obstacle infinitely small
    * if the robot is fully straight pointing to the obstacle. we find the angles and plot them in configuration space...
    * we draw the curve that connects these points in config space
* if the obstacle has size and is not just a point but a circle then the plot of points in config space is not just aline but a spape (like a thick line). all q1,q2 points in this area are illegal because they hit the obstacle

### 8.2 C-space visualization

* If we want to see how the Config space looks like for a 2D robot check out the [UNC applet](https://robotics.cs.unc.edu/C-space/)
* If we draw an obstacle in cartesian space we see the illegal areas in config space (C-Space) because they lead to collision
* if the robot is between obstacles in cartesian spaceand we want to get it out
* whate we have to do is to find away between the illegal areas in C-SPACe
* if we keep adding obstacles in cartesian space we might trap the robot representation in C_SPACEs illegal areas
* the C-Space of a robot is multidimensional e.g 6D for 6 joint robot

### 8.3 Motion planning- arms vs. mobile robots

* a robot arm that needs to go from one place to other with presence of obstacles and a mobile robot that wants to go from A to B in presence of obstacles are very similar problems if we think the robot arms C-SPACE representation. both cases we seek for a path in a map. for mobile robot in cartesian space, for robot arm in C-Space
* How we store that map. there are 2 ways
    * polygonal. each obstacle is defined as polygon as a list of vertices that define it.
    * grid. a descrite map where for each cell we set if its free or not. also we mark the start and end cell. the entire grid is stored in memory, its a dence representation. Memory Greedy CPU friendly
* Motion Planning: Robot Arms
    * High Dimensional C-Space (often 6D)
    * Discretizing map into grid is not tracktable
    * Polygonal C-space obstacle map very hard to compute
    * knowing "where" you are on the map is generaly easy
* Motion Planning: Mobile Robots
    * Low dim C-Space (often 2D)
    * Discretizing map into grid is tracktable and often done
    * Polygonal obstacle map can be available (e.g floorplan)
    * Knowing "where" you are on the map is generaly hard
* 2D grid maps ofr mobile robots fit easily in mem. 6d C-SPAce grids for preision arms are difficult to fit. so discretising the grid for a robot arm with high precision is not tractable computationaly
* IN C-Space if we have an obstacle in cartesian space in C-space it looks like an area of strange shape. in 6D space we cannot percieve it
* The easy thing with robot arms is thats easy to know where we are at any moment. robots have joint encoders so we know joint angles so we know position of end effector
* for a robot navigating in space to know its position is not trivial. it uses sensors (LIDAR,Machine Vision)

### 8.4 Motion planning for robot arms, sampling-based algorithms

* Assume a robot arm with walls in 2 axis and a table in front of it
* wassume we have the models of obstacles in task (cartesian) space
* mapping obstacles from task space to joint space is hard
* answering point queries (is this point in C-Space $q=[\frac{\pi}{3},\frac{\pi}{5}]$ in collision?) is easy. INV KINEM => check if mesh of robot intersect the obstacle in cartesian space
* So we might not know the sHape of illegal area in C-Space but we can check points one by one by moving from C-Space to Task Space
* There are motion planning algorithms that use this approach (Sampling based Motion Planning Algotithms) aka Stochastic Motion Planning Algorithms
    * Search C-Space in random fashion
    * Take advantage of ability to check collisiona at any point we want
    * Random exploration is surprisingly powerful, especially in high dimensional. it allows to be unstuck if we get stuck in a dead end

### 8.5 Rapidly-exploring Random Trees (RRT)

* Rapidly-exploring Random Trees (RRT)
* Input: start and goal point in C-space
* Output: path from start to goal
* Algotithm: 
    * insert start point in tree
    * while tree can not connect to goal:
        * sample random point r in C-space
        * find point p in tree that is closest to r
        * add branch of predicted length from p in direction of r
        * if new branch intersects obstacle:
            * discart new branch (or shorten)
    * compute path from start to goal through tree
    * shortcut path: for any two points in path. add direct line unless direct line intersects an obstacle
* Note: many variations are possible: production level implementation have additional subtleties
* We draw a 2D workspace q1,q2 size 10x10
* we set a start point q1=2,q2=4, end point is q1=9,q2=8, and a rectangular obstacle between (4,4) and (6,7) and another between (2.5,0) and (4,1) and another between (1,8) and (3,9)
* we use python to generate random tuple between 0 and 10.
    * first is 7,7. is OK. we draw a branch for the tree starting from S towards 7,7 for predefined length (1) and continue. the end point is A
    * second random point is (1,7). the closest tree point is S. so we draw a branch from S to the new point of predefined length, the end point is B
    * next point is (8,5). closest tree point is A. if i draw a line towards it in the predefined length i hit the obstacle. so we draw a shorter line. the end point is C
    * next point is (1,8) closest point is B. we draw the line at predefined length
* we continue the drill building the tree towards the Goal point
* we draw a direct line from a tree end point to goal to see if we can reach without hitting obstacle. if yes we are done
* the path is the branch from start to tree end point and then to goal
* after we get the path we attempt shortcuts by joining not adjacent points with straight lines. if it passes i throw away the long path
* if we keep start and gal points and change obstacles. making it harder. the tree will solve it and go to goal but it will take time to grow a tree that solves the problem
* in this algo we ask if lines intersect with obstacles.. with what we know so far we can ask the question for points not lines. so we need to discretize the line and sample points in it

### 8.6 Probabilistic Roadmaps (PRM) 

* Map construction:
    * While number of points in roadmap lower than threshold
        * sample *random* point in C-space
        * if new point is not in collision:
            * connect new point to all other points in the roadmap vis lines, as long as lines do not *intersect obstacles*.
* Input: start and goal point in C-space
* Output: path from start to goal
* Path finding:
    * connect start point to nearest point in roadmap such that connecting line does not *intersect obstacle*.
    * connect goal point to nearest point in roadmap such that connecting line does not *intersect obstacle*
    * *find a path between start and goal going exclusively on the roadmap*
* External calls in italics. Checking if a line intersects an obstacle is done by dicretizing the line. and then checking individual points for collision
* again a stochastic algo
* We draw a 2D workspace q1,q2 size 10x10
* we draw some random obstacle shapes.
* we get 10 random points
* we connect each one with as many from the others as possiblew with straight lines that dond hitting the obstacle
* we end up with a mesh (roadmap)
* we get the start and goal points, we do the same like before but in case of multiple connections we choose the shortest line
* if we have the roadmap ready anytime we get a goal and a start we use it to get the path
* for diffiult obstacles the paths might be separated so we cannot go from one side to the other
* Sampling based Motion Planning (Stochastic) - Recap:
    * only requires the ability to quickly check if a point in C-space is "legal" or not (often that means collision-free)
    * many versions are probabilisticaly complete. if a path exists it will be found in finite time...
    * ...but make no guarantees. in the worst case, time to solution can be very long (longer than exhastive search)
    * in practice these algorithm tend to be very effective in high-dimensional spaces
    * there are also no guarantees regarding quality of solution
        * not guaranteed to be the shortest possible
        * in practice, often needs post processing (to eliminate zig zag)

### 8.7 Motion Planning I- Demo and Recap

* SW package [moveit](http://moveit.ros.org) contains many ready implementations of motion planning algorithms
* we install it to run a demo.
* in catking workspace 'moveit_demo_workspace' we source `devel/setup.bash`
* we `roslaunch hw3 hw3.launch`
* we use a simulator of the baxter robot `roslauch baxter moveit config demo baxter.launch`
* baxter has 2 arms with 7DOF. we can plan for both arms in same time. what we end up is a 14DOF problem
* for simplicity we will pln for the left arm only. in planning group we select left arm
* we ask for new positions of the arm by dragging it and moving it. 
* moveit will plan a path and move the arm to go to the destination
* we add a scene object as an obstacle.
* we go to planning asking it to find a path by moving the arm in a position bypassing the obstacle
* sometimes the path is not optimal but it works
* Motion Planning Recap:
    * Generally formulated as the search of a path from start to goal through obstacles
    * mobile robots: direct application in task space
    * robot arm: articulated movement in task space becomes path through C-space (config space)
    * For dexterous arms C-space is high-dimensional, and translating obstacle outlines from task space to C-space is hard
    * Practical planning for arms: samplins-bassed algorithms
    * based on "random" exploration of C-space
    * only require the ability to check points in C-space

### Project 5

*Project 5 Description*
* In the previous project, you implemented a Cartesian controller for a 7-jointed robot arm. With this you could interactively control the position of the end-effector in Cartesian space. However, this method of control is not sufficient in the presence of obstacles. In order for the robot end-effector to reach a desired pose without collisions with any obstacles present in its 
environment we need to implement motion planning. In this project you will code up a Rapidly-exploring Random Tree (RRT) motion planner for the same 7-jointed robot arm. This will enable you to interactively maneuver the end-effector to the desired pose collision-free.

*The motion_plan(...) function*
* You are given starter code including the motion_planning package, which in turn contains the `motion_planning.py` file you must edit. Specifically, you must complete the motion_plan function. The arguments to this function and the methods provided by the MoveArm class are all you need to implement an RRT motion planning algorithm. 
* The arguments to the motion_plan function are:
    * q_start: list of joint values of the robot at the starting position. This is the position in configuration space from which to start
    * q_goal: list of joint values of the robot at the goal position. This is the position in configuration space for which to plan
    * q_min: list of lower joint limits
    * q_max: list of upper joint limits
* You can use the provided `is_state_valid(...)` method to check if a given point in configuration space is valid or causes a collision. The motion_plan function must return a path for the robot to follow. A path consists of a list of points in C-space that the robot must go through, where each point in C-space is in turn a list specifying the values for all the robot joints. It is your job to make sure that this path is collision free.
*Algorithm Overview*
* The problem we are facing is to find a path that takes the robot from a give start to another end position without colliding with any objects on the way. This is the problem of motion planning. The RRT algorithm tackles this problem by placing nodes in configuration space at random and connecting them in a tree. Before a new node is added to the tree, the algorithm also checks if the path between them is collision free. Once the tree reaches the goal position, we can find a path between start and goal by following the tree back to its root. 
* The algorithm follows the steps presented in detail in the lecture on Rapidly-exploring Random Trees.  Let us break this task into smaller steps:
* Create an RRT node object. This object must hold both a position in configuration space and a reference to its parent node. You can then store each new node in a list
* The main part of the algorithm is a loop, in which you expand the tree until it reaches the goal. You might also want to include some additional exit conditions (maximum number of nodes, a time-out) such that your algorithm does not run forever on a problem that might be impossible to solve. In this loop, you should do the following:
    * Sample a random point in configuration space within the joint limits. You can use the random.random() function provided by Python. Remember that a "point" in configuration space must specify a value for each robot joint, and is thus 7-dimensional (in the case of this robot)!
    * Find the node already in your tree that is closest to this random point.
    * Find the point that lies a predefined distance (e.g. 0.5) from this existing node in the direction of the random point.
    * Check if the path from the closest node to this point is collision free. To do so you must discretize the path and check the resulting points along the path. You can use the is_state_valid method to do so. The MoveArm class has a member q_sample - a list that defines the minimum discretization for each joint. You must make sure that you sample finely enough that this minimum is respected for each joint.
    * If the path is collision free, add a new node with at the position of the point and with the closest node as a parent.
    * Check if the path from this new node to the goal is collision free. If so, add the goal as a node with the new node as a parent. The tree is complete and the loop can be exited.
* Trace the tree back from the goal to the root and for each node insert the position in configuration space to a list of joints values.
* As we have been following the branches of the tree the path computed this way can be very coarse and more complicated than necessary. Therefore, you must check this list of joint values for shortcuts. Similarly to what you were doing when constructing the tree, you can check if the path between any two points in this list is collision free. You can delete any points between two points connected by a collision free path.
* Return the resulting trimmed path.

*Setup*
* As in the previous projects, please make sure that the first thing you do in your workspace is to source the 'setup_project5.sh' script. This starts up the simulated robot and the interactive controls you can use to move it. You can now press the 'Connect' button and you should see both the robot and the controls. 

*Grading/ Debug*
* After you have implemented the RRT algorithm, you can run your code with rosrun motion_planning motion_planning.py and use the interactive controls. You can move the controls around in space, however the robot will not immediately follow as it did in the last project. Instead, right clicking on the controls will open a menu, in which you can command the arm to move to the desired position. You can also add obstacles of varying complexity or run a grader we provide for you to test your code. This will test your implementation on all three objects and tell you if there have been any collisions. It gives your algorithm 10, 30 and 120 seconds for each object in order of increasing complexity. 
* Keep in mind that the RRT algorithm is stochastic in nature. That means that it will have different results every time you run it. Therefore, it is possible that the algorithm finds a path within the time given one time and times out another time. Particularly for the most complex obstacle running time can vary considerably. The same of course is true for the grade you get when you press the 'Submit' button. We encourage to test with the grader we provide for you until you get consistent results. The delay between you submitting and receiving a grade can be a little longer than in the previous projects, but should be no longer than 5 minutes.

### Project 5 FAQ

*How do run this project in my own Ubuntu machine?*
* 1) Launch Project 5, then in Vocareum click Actions>Download Starter code. This will download all the files you need to make the project run locally in your computer.
* 2) Install the needed ROS package(s). Run the following lines on your terminal:
```
sudo apt-get update
sudo apt-get install python-wstool ros-kinetic-moveit*
```
* Replace kinetic with the ROS version that you are running on your local machine.
* 3) IGNORE all the files other than the `catkin_ws` folder and `kuka_lwr_arm.urdf` file. Put the `catkin_ws` and the `kuka_lwr_arm.urdf` file in your home directory. 
* 4) The downloaded files are structured as a catkin workspace. You can either use this structure directly (as downloaded) and build the workspace using the "catkin_make" command or use whatever catkin workspace you already had, and just copy the packages inside your own src folder and run the catkin_make command. If you are having troubles with this, you should review the first ROS tutorial "Installing and configuring your ROS Environment".
* 5) Once you have a catkin workspace with the packages inside the src folder, you are ready to work on your project without having to make any changes in any of the files. Navigate to the catkin workspace folder and build the workspace using the command "catkin_make".
* 6) NOTE: You can source both your ROS distribution and your catkin workspace automatically everytime you open up a terminal automatically by editing the ~/.bashrc file in your home directory. For example if your ROS distribution is Kinetic, and your catkin workspace is called "project5_ws" (and is located in your home directory) then you can add the following at the end of your .bashrc file:
```
source /opt/ros/kinetic/setup.bash
echo "ROS Kinetic was sourced"
source ~/project5_ws/devel/setup.bash
echo "project5_ws workspace was sourced"
```
* This way every time you open up a terminal, you will already have your workspace sourced, such that ROS will have knowledge of the packages there.
* 7) Before moving forward, if you haven't followed the instructions on step 6, you will need to source ROS and the catkin workspace every time you open a new terminal. To run the project, first open up a terminal and run `roslaunch motion_planning mp.launch`. In the second terminal, run `rosrun motion_planning marker_control.py`. Note that you do NOT have to run `roscore` as roslaunch includes all the necessary packages.
* 8) On another 2 separate terminals, run `rosrun motion_planning motion_planning.py` (this is what you need to edit to complete the project), and as always `rosrun rviz rviz` to visualize the robot.
* 9) On rviz, you will need to add a RobotModel, InteractiveMarker, and Marker. When you add the InteractiveMarker, click on "InteractiveMarker" to expand it, and select /control_markers/update as the update topic. You shouldn't need to do anything for the RobotModel and the Marker. 

*Pyassimp error with kinetic*
* Edit "/usr/lib/python2.7/dist-packages/pyassimp/core.py" as follow :
```
-    load, load_mem, release, dll = helper.search_library()
+    load_mem, release, dll = helper.search_library()
```
* (You should navigate to this path and use sudo gedit core.py to edit the file. Once its open find the mentioned line (-) and replaced it with the one marked (+))
* If it doesn't work, try to update the pyassimp module to latest 3.3 version.
```
sudo pip -H uninstall pyassimp
sudo pip -H install pyassimp
```

*XML error running roslaunch locally on kinetic ubuntu*
* You have to edit the file `lwr_robot/lwr_defs/defs/util_defs.xml` and ensure matching parantheses in the marcro definitions as following :
```
<?xml version="1.0"?>
<robot xmlns:sensor="http://playerstage.sourceforge.net/gazebo/xmlschema/#sensor"
       xmlns:controller="http://playerstage.sourceforge.net/gazebo/xmlschema/#controller"
       xmlns:interface="http://playerstage.sourceforge.net/gazebo/xmlschema/#interface">

  <property name="M_PI" value="3.1415926535897931" />

  <!--
     Little helper macro to define the inertia matrix needed
     for links.
     -->
  <macro name="cuboid_inertia_def" params="width height length mass">
    <inertia ixx="${(mass * (height * height + length * length) / 12)}"
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
$$g(r)=min[g(r),g(n)+d(n,r)]$$

* in this algo at anytime we keep track of the shortest path from start to a set of parrticular nodes
* all nodes are unvisited at first
* we know only start node and length to itself is 0: $g(S)=0$ we mark it as visited
* we look at S neighbours. the weight of the node is the node N1 is the g of S + the weight of the edge s->N1 so g(N1)=11 . similarly g(N2)=10. we mark N2 as visited because it has lowest weight. we look to all its unvisited neighbours N1,N3,N4 => g(N3)=21, g(N4)=42 g(N1)=28. because what we have for N1 is lower. lowest marked node is N1 with 11 so we mark it as visited
* we look at unvisited neighbors of N1 (N8,N3)
* we repeat till we visit the goal. th epath length is 58
* Dijksta's Algorithm
* for any node n, keeps track of the *length of the shortest path from start to n found so far*, labeled $g(n)$
* Key Idea: visit closest nodes first
* Guarantee: once a node n has been "visited", $g(n)$ is equal to the length of the shortest part that exists from S to n.
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

* For any node n, also uses a *heuristic that estimates how far n is from the goal*, labeled here $h(n)$
* A heuristic is *admissible* only if it never over-estimates the real distance. A commonly used hevristic that meets this requirement is straight-line distance to goal
* Input: visibility graph, start S, goal G
* Output: path from S to G
* Algorithm: 
* mark all nodes "unvisited"
* mark S as having $g(S)=0, f(S)=g(s)+h(s)$
* while unvisited nodes remain:
    * choose unvisited node n with lowest $f(n)$
    * mark n as visited
    * for each neighbor r of n:
$$g(r)=min[g(r),g(n)+d(n,r)]$$
$$f(r)=g(r)+h(r)$$

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
$$ω(R-\frac{l}{2})=V_R$$
$$ω(R+\frac{l}{2})=V_L$$
$$R=\frac{l}{2}\cdot \frac{V_R+V_L}{V_L-V_R}$$
$$\omega=\frac{V_L+V_R}{l}$$

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