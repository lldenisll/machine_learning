##### Web app developed using python and streamlit (mostly). This web application allows the users to choose the classification algorithm they want to use and let them interactively set hyper-parameter values.
#### Classification algorithm availables:
- Suport Vector Machine (SVM) 
	- Regularization Parameter (c)
	- Kernel functions:
		- rbf or linear
	- Gamma
		- Scale or Auto
		
- Logistic Regression
	- Regularization Parameter (c)
	- Maximum number of iterations
	
- Random Forest
	- The number of trees in the forest
	- Maximum depth of the tree
	- Bootstrap samples when building trees
		- True or False



#### Requirements:
- streamlit
- pandas
- numpy
- sklearn
- Python3
