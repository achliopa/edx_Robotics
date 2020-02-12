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
    * Saviok
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

* 