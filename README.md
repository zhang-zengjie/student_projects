# Student Projects

## 1. Safe On-Ramp Merging for Autonomous Vehicles with Automated Formations

#### Background

On-ramp merging is an important problem popularly studied in autonomous driving and traffic control [1]. It is concerned with how to safely merge vehicles on the ramp to the traffic flow on the highway. The primary concern of the conventional on-ramp merging method is as essential as simply collision avoidance, i.e., maintaining a certain distance between the vehicles. However, more requirements are raised besides collision avoidance as more vehicles are incorporated into the on-ramp merging problem. For example, the merging of a platoon of vehicles requires that the vehicles maintain a certain formation while avoiding collisions with the vehicles on the highway traffic flow [2]. In this sense, the control problem may become very complicated, and high-level decision-making must be incorporated.

Instead of designing complicated hierarchical frameworks for both high-level decision-making and low-level control, in this project, we solve the complicated on-ramp merging problem by synthesizing a symbolic controller for a set of temporal logic (TL) specifications that encode both requirements for safe merging and platooning. A quadratic-planning-based method will be used to refine the symbolic controller such that the TL specifications are strictly satisfied. 

#### Task Description

The project is intended for one master student who has a control background and is interested in robotics or autonomous driving. You should have programming experience in either Python or MATLAB. In this project, you are expected to:

1. Read the literature and explore more related work on the multi-vehicle on-ramp merging problem.
2. Formulate the on-ramp merging problem mathematically and determine several use cases.
3. Propose a framework and the corresponding methods to solve the problem using TL specifications.
4. Visualize the use cases in the PreScan simulation environment.
5. Write the report.

To boot up this project, we will provide a Python program used to solve a similar problem and several related papers on on-ramp merging or on general autonomous driving problems. During this project, you will learn:

1. Ideas and concepts on autonomous driving and formal control
2. Using Python or MATLAB toolboxes to solve a synthesis problem for TL specifications
3. Visualizing autonomous driving results in a simulation environment

#### Reference

[1] el Abidine Kherroubi, Zine, Samir Aknine, and Rebiha Bacha. "Novel decision-making strategy for connected and autonomous vehicles in highway on-ramp merging." IEEE Transactions on Intelligent Transportation Systems 23.8 (2021): 12490-12502.

[2] Wang, Ziran, et al. "Cooperative ramp merging system: Agent-based modeling and simulation using game engine." SAE International Journal of Connected and Automated Vehicles 2.2 (2019).



## 2. Learning Temporal Logic Specifications from Human Driving Demonstrations
(In cooperation with Uppsala University)

#### Background

Learning to drive is a primary step toward obtaining a driving license. However, teaching an autonomous vehicle to drive can be far more difficult than teaching a human since a vehicle does not automatically recognize high-level driving patterns out of low-level commands. For example, the stopping behaviors for a red traffic light and for a crossing pedestrian mean no difference for the control of a vehicle. The conventional imitation learning or reinforcement learning approaches can only enable the vehicles to learn low-level behaviors, such as whether to turn left or right or stop depending on how far a vehicle is away from the roadside. On the contrary, high-level driving patterns can not be learned such as what to do when seeing pedestrians or different traffic lights. This project is intended to propose a novel framework for the imitation learning of autonomous driving. The high-level driving patterns will be formulated as parameterized temporal logic (TL) specifications and will be learned from human driving demonstrations.

#### Task Description

This project is intended for a master student who is interested in solving autonomous driving problems using learning and graph optimization approaches. We will collaborate with Uppsala University, Sweden, on human data collection and management. You are expected to represent the high-level patterns of autonomous driving as a temporal logic specification which is formulated as a parameterized directed acyclic graph (DAG) [1], such as a decision tree [2, 3]. You will also use the driving kit and the simulation environment in the lab to generate human driving demonstrations. Then, you learn the parameters and the structure of the DAG such that the human demonstrations satisfy the trained TL specification to the highest extent [4]. Your specific tasks in this project include:

1. Reproduce the TL learning method in the previous work;
2. Collaborate with Uppsala University for human demonstration recording;
3. Learn driving specifications from the recorded human demonstration;
4. Write the report.

To boot up the project, we will provide a simulation environment and a driving kit for human data collection. Uppsala University will help us deal with the ethical issues. Through the project, you will learn:

1. How to create a dataset from human experiments;
2. Structural representation of TL specifications;
3. The most cutting-edge technology in research: learning TL specifications.

#### Reference

[1]. Puranic, Aniruddh G., Jyotirmoy V. Deshmukh, and Stefanos Nikolaidis. "Learning from demonstrations using signal temporal logic in stochastic and continuous domains." IEEE Robotics and Automation Letters 6.4 (2021): 6250-6257.

[2]. Bombara, Giuseppe, and Calin Belta. "Offline and online learning of signal temporal logic formulae using decision trees." ACM Transactions on Cyber-Physical Systems 5.3 (2021): 1-23.

[3]. Bombara, Giuseppe, and Calin Belta. "Online learning of temporal logic formulae for signal classification." 2018 European Control Conference (ECC). IEEE, 2018.

[4]. Chou, Glen, Necmiye Ozay, and Dmitry Berenson. "Learning temporal logic formulas from suboptimal demonstrations: theory and experiments." Autonomous Robots 46.1 (2022): 149-174.

## 3. Risk-Awareness Approximation for Autonomous Driving Using Gaussian Process

#### Background

The next generation of autonomous driving requires that the vehicle not only be aware of its current situation in the environment but also has a decent prediction about what is going to happen in the short future. If the vehicle is aware of the potential risks ahead, it has a larger likelihood to take reactions in advance to avoid the occurrence of risky events. Linear/Signal temporal logic (LTL/STL) provides an effective tool to evaluate the probability of risky events for an autonomous vehicle with a stochastic dynamic model. This probability is indeed a function of the current state, the successive state, and the current action of the vehicle. It is recognized as a metric of risk awareness.

However, an analytical form does not exist for the probability function as mentioned above. Also, it is very difficult to address the continuity of the risk probability in the domain of the system state. As a result, the gradient of the risk-awareness metric, which is important for optimization solutions, is difficult to solve or even may not exist. This brings up a big challenge to solving risk-mitigating controllers using optimization-based methods.

#### Task Description

This project aims at approximating the analytical form of the risk probability function using Bayesian optimization (BO). As the most popular variant of BO, the Gaussian process (GP) is strongly suggested since there exist many toolboxes to train GP models. You will follow the previous work [1] to set up the dynamic sampling process and define driving specifications. Then, you will use BO/GP to fit the sampled data. Your tasks include:

1. Set up a simulation environment with dynamic sampling, following [1];
2. Translate the risk metrics in [1, 2] to STL formulas;
3. Use BO/GP to learn the risk probability, following similar work in [1].
4. Use the GP-based risk probability to design a risk-mitigating decision-maker and test it in a simulation environment PreScan.
5. Write the report.

Through this project, you will learn:

1. How to use BO/GP to learn an unknown function.
2. How to perform dynamic sampling for autonomous driving.
3. How to visualize your results in a simulation environment. 

#### Reference

[1]. Ramakrishna, Shreyas, et al. "Risk-aware scene sampling for dynamic assurance of autonomous systems." 2022 IEEE International Conference on Assured Autonomy (ICAA). IEEE, 2022.

[2]. Dang, Ni, et al. "Combining Event-Based Maneuver Selection and MPC Based Trajectory Generation in Autonomous Driving." Electronics 11.10 (2022): 1518.

## 4. Understanding Target Intentions to Mitigate Driving Risks
(In cooperation with TUM)

#### Background

The next generation of autonomous driving is looking forward to three additional capabilities than the current. 1. The ego vehicle is able to predict the likelihood of risks in its short future; 2. The ego vehicle is able to understand or tries to understand the intentions of its target vehicles. 3. The ego vehicle incorporates stochastic environmental uncertainties into the prediction and understanding of risks and target intentions. For a vehicle represented as a stochastic dynamic model, the likelihood of risks can be characterized by a signal temporal logic (STL) specification that is concerned with essential driving metrics. Then, the understanding of target intentions is based on the prediction of the target behaviors based on the observation of the actions of the targets. In this sense, the controller design of autonomous vehicles can be formulated as a model predictive control (MPC) problem. The solution is expected to have better capabilities of avoiding the occurrence of risky events.

#### Task Description

This project is intended for a master student who is interested in solving autonomous driving control problems using STL and model predictive control methods. The main objective of the project is to propose a novel risk-and-intention-aware (RIA) autonomous driving framework as mentioned above and test it in a simulation environment. Your tasks include:
1. Explore related work on MPC and autonomous driving;
2. Translate the risk metrics in [1, 2] to STL formulas;
3. Use the methods in [3] for the understanding of target intentions;
4. Solve the MPC problem with STL specifications and target intentions.
5. Write the report.

Through this project, you will learn:

1. Knowledge and applications of autonomous driving;
2. How to use model predictive control to solve an autonomous driving controller subject to STL specifications;
3. Visualize your results in a simulation environment.

#### Reference

[1]. Ramakrishna, Shreyas, et al. "Risk-aware scene sampling for dynamic assurance of autonomous systems." 2022 IEEE International Conference on Assured Autonomy (ICAA). IEEE, 2022.

[2]. Dang, Ni, et al. "Combining Event-Based Maneuver Selection and MPC Based Trajectory Generation in Autonomous Driving." Electronics 11.10 (2022): 1518.

[3]. Dang, Ni, et al. "Distributed Stochastic Model Predictive Control for a Microscopic Interactive Traffic Model." Electronics 12.6 (2023): 1270.

## 5. Using ChatGPT to Interpret Natural Language as Temporal Logic Specifications

#### General description

ChatGPT is a world-changing tool that facilitates communication between humans and artificial intelligence. For most questions formulated in natural language, ChatGPT can provide helpful hints leading to the answers to these questions, even though the answers are not validated and may be wrong.

#### Task description

This project aims at developing a framework using ChatGPT to interpret human natural language into temporal logic (TL) specifications that can be understood by robotic systems. Note that the TL specifications given by ChatGPT may be incorrect or even infeasible. The student is expected to a development pipeline to automatically check the correctness of the generated ChatGPT and provide the results to the user in an intuitive manner for improvement. The algorithm should also give likely suggestions to tell the users how to improve the natural language input. What's more, a feedback loop is needed to infer the performance of the deployment of the TL specifications on the robot system and provide useful suggestions accordingly. A use case of motion planning for a delivery robot is needed to validate the framework.

## 6. Robot Motion Planning with Temporal Logic Specifications Using Continuous Tensor Decomposition

#### General Description

Motion planning and control are important problems for most robotic applications. Conventional robot planning problems are concerned with the achievement of an ultimate goal and safety requirements. However, the next generation of automatic robotic systems is expected to satisfy more complicated specifications, such as avoiding risky events or ensuring dependence among different subtasks. These complicated specifications can be formulated as temporal logic (TL) formulas. Then, the robot motion planning problem can be formulated as finding an optimal policy to maximize the likelihood of specification satisfaction.

#### Task Description

This project aims at using reinforcement learning to solve the robot motion planning problem subject to TL specifications [1-3]. You are expected to use the technology of tensor decomposition to reduce the rank of the problem to avoid the "curse of dimensionality" [4]. You are expected to analyze the efficacy and convergence of the method and validate the method in a quadcopter simulation use case.

[1]. Kapoor, Parv, Anand Balakrishnan, and Jyotirmoy V. Deshmukh. "Model-based reinforcement learning from signal temporal logic specifications." arXiv preprint arXiv:2011.04950 (2020).

[2]. Hasanbeig, Mohammadhosein, Daniel Kroening, and Alessandro Abate. "Deep reinforcement learning with temporal logics." Formal Modeling and Analysis of Timed Systems: 18th International Conference, FORMATS 2020, Vienna, Austria, September 1–3, 2020, Proceedings 18. Springer International Publishing, 2020.

[3]. Li, Xiao, Cristian-Ioan Vasile, and Calin Belta. "Reinforcement learning with temporal logic rewards." 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017.

[4]. Gorodetsky, Alex, Sertac Karaman, and Youssef Marzouk. "High-dimensional stochastic optimal control using continuous tensor decompositions." The International Journal of Robotics Research 37.2-3 (2018): 340-377.

## 7. Efficient Task Assignment for Multi-Agent Systems
(In cooperation with KTH)

#### Background

Cooperation is always seen in human society where several players are needed to accomplish a series of interdependent tasks jointly. A popular game named "Overcooked" demonstrates how several players, typically two, collaborate to cook soup, hamburgers, french fries, and pasta and serve them to the guests with strict time limits. Efficient task assignment is the key to winning the game, which prescribes at what time, which player should do what. Similar examples are often seen in real life, especially in industrial manufacturing plants, hospitals, or restaurants. Nowadays, it is more and more demand for cooperative robots to accomplish these tasks.

This project is intended to investigate a framework of efficient task assignment for a multi-agent system. We will formulate the problem from a typical use case in the "Overcooked" game, where several robots are required to accomplish a series of cooking tasks given in a certain order. The tasks are specified using linear temporal logic (LTL) formulas with timing bounds [1]. The objective is to associate these LTL formulas with the corresponding robots such that the overall LTL satisfaction metric is maximized. Such a process renders a planning problem that drastically increases when the number of robots gets larger. Safe-critic controllers such as funnel function (FF) or control barrier function (CBF) can be used to limit the reachability of the agents in order to reduce the searching space of the problem [2]. The methods will be tested and evaluated in the Nvidia Issac-Sim simulation environment which provides convincing results close to reality.

The main challenge of this project is the technical point of formulating the task assignment problem as a dynamic planning problem or a decision-making problem [3]. The state of the dynamic planning problem does not only contain the kinematic state of the robots (position, velocity, etc), but also the accomplishment state of the tasks quantized as the satisfaction of the LTL formulas. The state transition of the planning model is governed by the assignment action from a finite set of decision variables. The solution is the optimal policy prescribing the mapping between states and assignments subject to an overall task cost [4]. A variety of off-the-shelf tools, such as Yalmip or Gurobi, can be used to solve the formulated planning problem.


#### Task Description

This project is intended for one master student who is interested in solving a concrete multi-robot motion planning problem and implementing the methods in a high-fidelity simulation environment. Your tasks in this project include:

1. Explore the up-to-date literature on task assignments for multi-agent systems.
2. Accomplish the formal formulation of task assignments as a dynamic planning problem.
3. Solve the problem using optimization solvers.
4. Reduce the complexity of the problem using heuristic controllers (FF-, CBF-based, etc).
5. Validate the methods in an abstract "Overcooked" use case.
6. Demonstrate the use case in the Issac-Sim simulation environment.

Through this project, you will learn:

1. How to formulate a dynamic planning problem from a real-life problem.
2. How to solve a planning problem using off-the-shelf tools.
3. How to use heuristics to simplify a planning problem.
4. How to generate high-quality robot demonstration in a high-fidelity simulation environment.

#### Reference

[1]. Leahy, Kevin, et al. "Scalable and robust algorithms for task-based coordination from high-level specifications (ScRATCHeS)." IEEE Transactions on Robotics 38.4 (2021): 2516-2535.

[2]. Liu, Siyuan, et al. "Compositional synthesis of signal temporal logic tasks via assume-guarantee contracts." 2022 IEEE 61st Conference on Decision and Control (CDC). IEEE, 2022.

[3]. Liu, Hong, et al. "A novel approach to task assignment in a cooperative multi-agent design system." Applied Intelligence 43 (2015): 162-175.

[4]. Salman, Ayed, Imtiaz Ahmad, and Sabah Al-Madani. "Particle swarm optimization for the task assignment problem." Microprocessors and Microsystems 26.8 (2002): 363-371.

## 8. Understanding Opponent Intention Using Branching-Time Temporal Logic


#### Background

In a two-player game, it is important to make predictions for what is going to happen in the next few steps. The prediction is more reliable when taking into account the possible reactions of the opponent. As an example, consider two autonomous cars that are driving along the highway, each of which intends to overtake another while trying to avoid collisions. If each car is able to precisely estimate whether its opponent is going to overtake it, it can more easily make the decision whether to overtake or maintain the current state. By "more easily" we mean that the decision can be made with less uncertainty.

The possible decisions of the opponent player can be formulated as a decision tree of which the root is the current state of the opponent [1]. In this sense, the possible future states of the opponent can be included in a computational-tree logic (CTL) formula in which each branch corresponds to a collision risk [2]. Therefore, the ego player can trust the most-possible branch of the opponent's CTL corresponding to the smallest risk, so as to make the decision incorporating the trusted intention of the opponent.

#### Task Description

This project is intended for a master student who is interested in using branching-time temporal logic (TL) to estimate an opponent's intention to facilitate autonomous driving. You are expected to propose a novel framework for using CTL to depict the opponent's intention and incorporate this information into the ego's decision-making. The methods are to be evaluated in MATLAB and validated on two robot cars. Specifically, Your tasks include:

1. Explore related literature on intention estimation for robot decision-making, especially autonomous driving.
2. Follow [1] to define the decision tree for the opponent car.
3. Transform the decision tree of the opponent car into a CTL formula.
4. Design the maneuver planner that minimizes the expected risk provided by CTL verification.
5. Evaluate the results using MATLAB.
6. Deploy the methods to robot cars.
7. Write the report.

Through this project, you will learn:

1. How to build up decision trees to solve robot decision-making problems.
2. How to use branching-time TL to estimate robot intentions.
3. How to use MATLAB to evaluate the performance of autonomous driving controllers.

#### Reference

[1]. Hu, Haimin, et al. "Active Uncertainty Reduction for Safe and Efficient Interaction Planning: A Shielding-Aware Dual Control Approach." arXiv preprint arXiv:2302.00171 (2023).

[2]. Penczek, Wojciech. "Branching time and partial order in temporal logics." Time and Logic. Routledge, 2019. 179-228.
