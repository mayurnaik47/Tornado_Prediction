# Tornado_Prediction
web system which reports the probability of tornado occurrence at the location requested by the user and generates an alert at that particular location using real-time user tweets.



Instructions :
Installation of Flask and creation of virtual environment.
1. Open command propmt and install flask and virtual environment by giving the command: 
	pip install flask
	pip install virtualenv.
2. Then, create a virtual environment by using the command: 
	virtualenv ENV
3. Now, go to the ENV directory you just created. 
4. From there, go to Scripts folder and give this whole path in the command prompt.
5. Now, give a command to activate the virtual environment:
	 activate
6. Now, follow the instructions for starting mongo server.

For starting MongoDB server:
1. Open command prompt and run mongoDB server by giving the command: mongod
2. Then, follow the steps for starting main server.

For starting main server:

1. From another command prompt, open go to "app" folder.
2. Activate the Virtual Environment as suggested in step 5 above.
3. Then, run tornado.py file by giving the command: python tornado.py from app folder.
4. After starting the server, go to the app folder and open templates folder. 
   Inside the templates folder, open home.html file and take a look at overall project execution.  
5. Enter the location for which you need to know the Tornado Prediction.
6. Navigate through application as described in word document.


For executing project:
1. Open home.html file, you will see a text box for entering your location. Type your location in it and click on the
   submit button.
2. After submitting your location, next page will show the probability of occurring tornado in the given location.
3. You can also generate an alert message for the locations showing high tornado probabilities.
