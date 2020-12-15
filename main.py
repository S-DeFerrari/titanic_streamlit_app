import sklearn
import pickle
import streamlit as st
from PIL import Image
import numpy as np

rf_model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

header_pic = Image.open('header.jpg')
st.image(header_pic, use_column_width=True)
st.title("Would You have Survived the Titanic?")

# Side bar portion of code
author_pic = Image.open('stephen.jpg')
st.sidebar.image(author_pic, "Your humble app creator", use_column_width=True)
st.sidebar.markdown("[Hello](https://github.com/S-DeFerrari)")
st.sidebar.write("This app is powered by Machine Learning!")
st.sidebar.write("It uses a Random Forest Classification model "
                 "trained with Kaggle's now legendary Titanic Survivor dataset. This model was correct 83.6% of the "
                 "time when it came to predicting whether a person made it onto a lifeboat or was lost"
                 " at sea.")
st.sidebar.write("I hope you enjoy this and remember:")
st.sidebar.write("Women and Children First!")

# Main block asking questions about user
name_st = st.text_input("What is your name?")

# Title section
title_list = sorted(['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Major', 'Col',
              'Mlle', 'Ms', 'the Countess', 'Capt', 'Jonkheer', 'Don', 'Dona', 'Sir', 'Mme', 'Lady'])
title_st = st.selectbox("Which title would you prefer?", options=title_list)

# Gender columns with portraits
male_pic = Image.open('male.jpeg')
female_pic = Image.open('female.png')
other_pic = Image.open('other.jpeg')

st.text("Please choose your gender:")
col1, col2, col3 = st.beta_columns(3)
with col1:
    st.image(male_pic, use_column_width=True)
    male = st.checkbox('Male')
with col2:
    st.image(female_pic, use_column_width=True)
    female = st.checkbox('Female')
with col3:
    st.image(other_pic, use_column_width=True)
    other = st.checkbox('Other')

# Age
age = st.number_input("What is your age?", value=1, step=1)

# SO
spouse = st.checkbox("You are in a relationship.")

# Children
children = st.number_input("How many children do you have?", value=0, step=1)

# Parents
parents = 0
if st.checkbox("You often travel with your parents."):
    parents = st.selectbox('Which ones?', options=['Just Mom', 'Just Dad', 'Usually both'])

# Siblings
siblings = st.number_input("How many siblings do you have?", value=0, step=1)

# Class
class_options = ['First Class', 'Business', 'Coach']
class_st = st.selectbox("When you last flew, which class were you in?", options=class_options)

# Fare - There isn't a way to do this well so I'm going to use the mean from the original for now
fare = 32

# Departure Point
departure = st.selectbox("Where would you prefer to visit?", options=['Britain', 'France', 'Ireland'])

# Cabin Number
cabin_options_list = ['Hanging with Friends', 'Eating a Nice Meal', 'Reading a Good Book',
                      'Taking a Nap', 'Working out at the Gym', 'Playing Organized Sports',
                      'Going for a Swim', 'Being out in Nature', "I don't like any of these"]

cabin = st.selectbox("Which of these activities do you prefer most?", options=cabin_options_list)

# Multiple Cabins
mult_cabins = st.checkbox("You prefer to have your own room when traveling, even if it costs more.")

# Numeric Ticket
num_tick = st.checkbox("You always buy something immediately when you want or need it.")


# Processing Functions
def parents_func(response):
    """This function will return how many parents actually travel with the user"""
    if response == 0:
        return 0
    if response == 'Just Mom' or 'Just Dad':
        return 1
    else:
        return 2


def class_decider(response):
    """This function will take the response given for class and convert it into a dummied list"""
    global class_options
    ending_list = [0, 0, 0]

    for i, r in enumerate(class_options):
        if r == response:
            ending_list[i] = 1

    return ending_list


def sex_decider(male: object) -> object:
    """This function will take the response for sexes and spit out the proper dummied list"""
    if male:
        return [0, 1]
    else:
        return [1, 0]


def embarked(response):
    "This function will convert the departure point responses to their proper dummied list"
    if response == "Britain":
        return [0, 0, 1]
    if response == "France":
        return [1, 0, 1]
    else:
        return [0, 1, 0]


def cabin_decider(response):
    """This function will convert the activity picked into its cabin equivalent in a dummy list"""
    global cabin_options_list
    ending_list = []

    for i in range(0, len(cabin_options_list)):
        ending_list.append(0)

    for i, cabin in enumerate((cabin_options_list)):
        if cabin == response:
            ending_list[i] = 1

    return ending_list


def title_decider(title):
    """This function will convert the title chosen by the user and spit out a dummy list for them"""
    global title_list
    ending_list = []

    for i in range(0, len(title_list)):
        ending_list.append(0)

    for i, t in enumerate(title_list):
        if t == title:
            ending_list[i] = 1

    return ending_list


# Final
if st.button("Click to discover your fate"):
    # Scale the numerics
    siblings = siblings + int(spouse)
    parents = parents_func(parents) + children
    num_array = np.array([age, siblings,parents,fare])
    num_array = num_array.reshape(1,-1)
    num_scaled = scaler.transform(num_array)

    # Build the list
    fate_list = [num_scaled[0][0], num_scaled[0][1], num_scaled[0][2], num_scaled[0][3], int(mult_cabins), int(num_tick)]
    fate_list.extend(class_decider(class_st))
    fate_list.extend(sex_decider(male))
    fate_list.extend(embarked(departure))
    fate_list.extend((cabin_decider(cabin)))
    fate_list.extend((title_decider(title_st)))
    prediction = rf_model.predict([fate_list])

    survived_pic = Image.open('Old_Rose.jpg')
    death_pic = Image.open('frozen_jack.jpg')

    if prediction == 1:
        st.image(survived_pic, use_column_width=True)
        st.write(f"Congratulations! {title_st} {name_st} would have survived the Titanic!")

    else:
        st.image(death_pic, use_column_width=True)
        st.write(f"I am sorry, {title_st} {name_st} would have been lost at sea.")
