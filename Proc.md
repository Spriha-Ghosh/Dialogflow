Last time, we made a titanic project with machine learning. We made the backend with flask and used an ML model processed from Jupiter notebook. Here, we also use that backend. In case, you didn’t read this, check the series.

Our aim is to make a chatbot for the titanic project. What we did in the frontend in the previous series, we will do the same but with a chatbot. Moreover, in future, we will extend this to an explainable chatbot which we will discuss later.

This is the final output of this series.


Here, we chat in the frontend (Angular) which uses a node backend to communicate(send and receive messages) with Dialogflow. Dialogflow uses a flask backend to predict the ML model.

Hence, the steps of this series are delineated below:
1. Create a chatbot architecture in Dialogflow
2. Attach the flask backend with Dialogflow to get the ML model prediction
3. Make a custom UI for the chatbot
4. Make a node backend to handle all the communication between custom UI and Dialogflow.
Also, use this node backend to communicate with the flask backend for getting the ML model prediction.

Check out our online demo
All the codes of this series will be found on git.

Let’s start with step 1:

If you don't have any idea at all about Dialogflow, it is highly recommended that you should get some idea about it.
First, create an agent in the Dialogflow. In our case, we named this ‘Titanic_agent’


Then add a response to the default welcome intent so that the user provides his/her name.


Create an intent to get all the input variables:


Add required parameter and their training phases:


Also, don’t forget to enable webhook for the newly created intent.


If you see in the ENTITY column of the ‘Action and parameters‘ the last two parameters don't have a system entity. Instead, we created our own custom entity.
Let's create two custom entities named ticket_embark and ticket_price.


Define their values and synonyms



Finally, you can test the chatbot in the Dialogflow platform itself.
Check the right side where it is written as ‘Try it now.


So far we have just created a very simple chatbot.
In the next part, we will use webhook to receive all the input variables and get the prediction from the flask backend and finally show the result to the user in our bot.



![image](https://github.com/user-attachments/assets/a5b2b4fe-75c2-4da9-811c-40a09a1d196d)
Any Machine Learning model is pretty much useless unless you put it to some real life use. Running the model on Jupyter Notebook and bragging about 99.99% accuracy doesn’t help. You need to make an end-to-end application out of it to present it to the outer world. And chatbots are one fun and easy way to do that.

Building chatbots has never been so easy. Google’s DialogFlow is an obvious choice as it’s extremely simple, fast and free! Before proceeding, try out the app for yourself first here!

The Flow
Now that you’ve tried it, coming to building the complete application, we’d be going over below steps:

Your Machine Learning Model (Iris, in this case)
The DialogFlow Chatbot which fetches inputs from user
A Flask app deployed on any public host which renders the request and response
A webhook call which your chatbot makes to the flask api to send the data and fetch the result
Integrating DialogFlow with Telegram
We’ll go over each step one by one. Let’s first take a look at how the architecture of our complete app will look:


The Architecture
What is happening?
So the user has access to the Telegram chatbot which we will be built on DialogFlow and integrate with Telegram later. The conversation starts and the chatbot prompts the user to input the Data, which are the flower dimensions (Petal length, Petal width, Sepal length and Sepal width). Once the chatbot receives the last input, it will trigger a webhook call to the flask API which will be deployed on a public host. This flask API consists of our app which will retrieve the 4 data points and fit that to our Machine Learning model and then reply back to the chatbot with the prediction. You can find the complete code at my Github.

Now let’s go over each step!

Building the components
The ML model
First, let’s build a basic ML model which take Iris dimensions and predicts the Iris type. No rocket science here. Just a very basic model which renders result with decent accuracy. Below is the bare-bones code to that quickly.

#Load data
iris = load_iris() 
X = iris.data      
y = iris.target

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.25)

#Define and fit to the model
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(accuracy_score(predicted, y_test))
print(clf.predict(X_test))

#Save the model as Pickle
import pickle
with open(r'rf.pkl','wb') as model_pkl:
    pickle.dump(clf, model_pkl, protocol=2)
We just load the data and fit it to Random Forest classifier. No need of cleaning the data as the dataset is already clean extremely small. I am not diving into any optimization here just to avoid complexity as our main aim is not the model accuracy but the complete application. Then just pickle the model and later this model, ‘rf.pkl’, will then be loaded in our flask app.

The DialogFlow Chatbot
Now let’s dive straight into DialogFlow to make our chatbot. You can use other APIs and frameworks as well to build a chatbot but Google’s DialogFlow is an obvious choice as its easy, free and super quick to build! Go to DialogFlow and sign in with your Google account. Then click on ‘Create Agent’ to create your chatbot.


Creating the agent
Next, we need to create an intent which will ask the user for data and make a webhook call. Let’s first edit the Default Welcome Intent to make it ask for a ‘Yes’ or ‘No’ from a user.


Now as soon as the user types ‘Yes’, DialogFlow should call another intent which will ask the user for inputs and store the data points in ‘Entities’. Here we are dealing with simple random numbers so we don’t need to create our custom Entities. DialogFlow has default Entities in place to handle such data. So we need to create a ‘Yes- FollowUp Intent’ for this intent because that intent will be called after a positive reply from the user.


Click on ‘Add follow-up intent’ > ‘Yes’. You can rename this intent to something else if you want. I will rename this to ‘IrisData’. Now we need to add the entities, which will hold the data received from the user. We will just use the default @sys.number entity here for all the 4 inputs. Make 4 different parameters for the 4 data points needed from user — Petal Length, Petal Width, Sepal Length, Sepal Width. Make sure to add the prompts as well to ask the user for inputs separately.


Adding the parameters

Adding the prompts
Train the model with a few inputs so that it knows what to expect. You can test the chatbot now on the right panel to check if it is performing accordingly.

Once done, you’ll need to enable fulfillment by ‘Enable webhook call for this intent’. By doing this, this particular intent will make a webhook call to our app deployed on public host, which is Heroku. We now need to build the flask app and deploy it on Heroku and then put the URL in the ‘Fulfillment’ tab which is available on the left side.

Flask app on Heroku
We now need to build our flask app which gets the webhook call from our chatbot, retrieves the data, then fits to the ML model (rf.pkl) and returns back the fulfillment text to DialogFlow with the prediction. Below is the code:

# Importing necessary libraries
import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin

# Declaring the flask app
app = Flask(__name__)

#Loading the model from pickle file
model = pickle.load(open('rf.pkl', 'rb'))


# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():


    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r
Once this is done, we need to process the fulfillment request from DialogFlow which is in JSON format to retrieve data. The fulfillment request looks something like this:


API Request Format
So we need to get into ‘queryResult’ >> ‘parameters’ >> ‘number’, ‘number1’, ‘number2’, ‘number4’. Once retrieved, we will dump these data points into an array and slam it to our model and get the prediction.

# processing the request from dialogflow
def processRequest(req):


    result = req.get("queryResult")
    
    #Fetching the data points
    parameters = result.get("parameters")
    Petal_length=parameters.get("number")
    Petal_width = parameters.get("number1")
    Sepal_length=parameters.get("number2")
    Sepal_width=parameters.get("number3")
    int_features = [Petal_length,Petal_width,Sepal_length,Sepal_width]
    
    #Dumping the data into an array
    final_features = [np.array(int_features)]
    
    #Getting the intent which has fullfilment enabled
    intent = result.get("intent").get('displayName')
    
    #Fitting out model with the data points
    if (intent=='IrisData'):
        prediction = model.predict(final_features)
    
        output = round(prediction[0], 2)
    
    	
        if(output==0):
            flowr = 'Setosa'
    
        if(output==1):
            flowr = 'Versicolour'
        
        if(output==2):
            flowr = 'Virginica'
            
        #Returning back the fullfilment text back to DialogFlow
        fulfillmentText= "The Iris type seems to be..  {} !".format(flowr)
        #log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return {
            "fulfillmentText": fulfillmentText
        }


if __name__ == '__main__':
    app.run()
Once this is done, we just need to deploy the code on the public host. I have chosen Heroku as again, it is easy, free and super quick! You just need to add below files to your new Github repository: the flask app, the model pickle file, a Procfile (this is very essential and helps Heroku locate the flask app), and a requirements text file which tells Heroku which all libraries and versions to pre-install to run the app correctly.

Just make a repository on your Github and go to Heroku. Create a ‘New App’ and ‘Connect’ your Github repository there. Once connected, just hit the deploy button and you are done!


Connecting Github repo to Heroku app
The Webhook call
On to the final step now. We now need to connect our deployed app to our chatbot. Just enter the URL on which your app is deployed and add ‘/webhook’ to it. Remember from the flask code above that the app is routed to ‘/webhook’. Just go to the ‘Fulfillment’ tab on the left panel in DialogFlow, enable ‘Webhook’ and just add the <your_app’s_URL>/webhook.


Enabling Webhook on DialogFlow
And we are done! (Don’t forget to click on save button!) You can test on the right panel by initiating a chat to test if the webhook request/response is working fine. You should get the fulfillment response back with the prediction.

Integrating with Telegram
Coming to the final step. Nothing much to do here as integrating web apps with DialogFlow is very easy. We first need to go to Telegram to generate a dummy bot there and generate its token. Search for ‘BotFather’ and click on ‘new bot’. It will ask you for the bot name. Enter any name as you wish. It will then prompt you to enter a username.

After you have done that, a token and a link for your bot will be generated there. Just copy that token and go to DialogFlow ‘Integrations’ panel on the left. Enable Telegram there, paste the token you just generated and click on start. That’s it! Now just head on to the Telegram bot link and try out the app!
