import os
import streamlit as st
from OpenAI_API_KEY import openai_api_key

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

os.environ['OPENAI_API_KEY'] = openai_api_key

parser = StrOutputParser()

student_interest_topic = 'baseball'
number_of_frqs = 2

learning_standards = '''
Learning standards summary:
"Draw evidence from literary or informational texts to support analysis, reflection, and research." \

Category 1 - Key Ideas and Details:
-"Refer to details and examples in a text when explaining what the text says explicitly and when drawing inferences from the text."
-"Determine the main idea of a text and explain how it is supported by key details; summarize the text."
-"Explain events, procedures, ideas, or concepts, including what happened and why, based on specific information in the text."

Category 2 - Craft and Structure:
-Determine the meaning of words or phrases in a text relevant to subject area.
-Describe the overall structure of ideas, concepts, or information in the text.

Integration of Knowledge and Ideas:
-Interpret information presented and explain how the information contributes to an understanding of the text in which it appears.
'''

generator_prompt_string = f'''
Generate {number_of_frqs} free response questions written at a 4th grade reading level related to {student_interest_topic}. \
The subject should have an introduction followed by free response questions.
Each free response question should include 2 parts: 1 - context that goes into a specific aspect of the subject and 2 - a question that is based on the context \
and a free response question that tests the student's understanding of the information you have presented. \
Include facts in the context that let a student demonstrate their critical reading skills.
The context should contain a few paragraphs.
The Question section of each free response question should not include any statements, if possible.  If there is additional context needed, include it in the Context section.

For each free response question, use the following format:

Introduction:
---
An introduction of the subject
---

Context passage:
---
A few paragraphs about the subject.
---

Question:
---
A free-response written question testing the student's understanding of the information in the paragraphs in the passage.
---

Here's an example set of FRQs:
---
Subject: Human Homes that Master the Weather

Intro:
From hot, dry deserts to windy, freezing steppes, the earth has extreme
climates. People who live in these harsh climates must adapt to them. Some
need shelters that keep them warm in cold temperatures. Others need
homes that protect them from the heat. Many homes use the natural
environment to help them feel comfortable.

FRQ 1 Context: 
1
The village of Matmata, Tunisia, lies on the edge of the Sahara Desert.
On a summer day, the sun bakes the land to 110 degrees F or more. But
nighttime temperatures may be as low as 40 degrees F. Moisture in the air
holds heat. Temperatures in Matmata drop because the dry desert air
cannot hold heat.
2
To escape the burning sun and the nighttime chill, people in Matmata
live in underground caves. Villagers dig large holes 20 feet into the ground.
Ramps or staircases lead down to these holes. e holes serve as courtyards
for attached underground rooms. Tunnels connect the rooms. People even
carve their furniture from the rock walls.
3
To escape the burning sun and the nighttime chill, people in Matmata
live in underground caves. Villagers dig large holes 20 feet into the ground.
Ramps or staircases lead down to these holes. e holes serve as courtyards
for attached underground rooms. Tunnels connect the rooms. People even
carve their furniture from the rock walls.
4
The underground caves stay a constant temperature. e sandstone
walls absorb the sun’s heat. ese thick walls stay warm throughout the
night. By morning they have cooled off. ey remain cool during the day
while they slowly absorb heat from the sun. At dusk, the cycle begins again 

FRQ 1 Question:
According to paragraphs 2 through 4 of the article “Human Homes that Master the
Weather,” how does the sun affect the way people live in Matmata, Tunisia? Use
two details from the article to support your response.
---
'''
generator_prompt = ChatPromptTemplate.from_template(generator_prompt_string)
generator_model = ChatOpenAI(model='gpt-3.5-turbo')
generator_chain = generator_prompt | generator_model | parser
#question = generator_chain.invoke({'student_interest_topc': student_interest_topic, 'number_of_frqs': number_of_frqs})
#print(question)


question = '''
Intro:
Baseball is a popular sport played around the world. It is a game that involves two teams of nine players each, taking turns batting and fielding. The objective is to score more runs than the opposing team by hitting the ball and running around the bases. Baseball is often referred to as "America's pastime" and has a long history dating back to the 18th century.

FRQ 1 Context:
1
In baseball, the pitcher plays a crucial role in the game. The pitcher stands on a raised mound in the center of the field and throws the ball towards the batter. The pitcher's main goal is to throw the ball in such a way that it is difficult for the batter to hit it. They can throw different types of pitches, such as fastballs, curveballs, and changeups, to keep the batter guessing. The speed and accuracy of the pitcher's throws can greatly impact the outcome of the game.

FRQ 1 Question:
How does the pitcher's role affect the game of baseball? Provide two examples from the passage to support your answer.
'''

student_answer = '''
As the player who throws the ball to the batter, the pitcher plays a crucial role.  This is because they’re the one who can strike the batter out.  Also, if they don’t pitch so it is difficult for the batter to hit, that can hurt the pitcher’s team.  They have to know how to use different pitches, like fastballs and curveballs.  How fast and accurately they throw also affects whether they win or lose.
'''

grader_prompt_string = f'''
Evaluate the student's answer to the question below based on the learning standards below. \

How to evaluate the answer:
Assign a score to each category within the learning standards.  Assign scores in the following way: \
-If the majority of the items nested in 3 dashes within each category are present in the student's answer, assign a score of 'great'. \
-If only some of the items nested in 3 dashes within each category are present in the student's answer, assign a score of 'good'. \
-If none of the items nested in 3 dashes within each category are present in the student's answer, assign a score of 'needs improvement'.

Format:
---
<Name of first category>: score
-Provide feedback including what the student did well based on this category's criteria and what the student could improve on.

<Name of second category>: score
-Provide feedback including what the student did well based on this category's criteria and what the student could improve on.


etc., until all categories have been represented
---

Students Answer:
---
{student_answer}
---

Learnding Standards:
---
{learning_standards}
---
'''

grader_prompt = ChatPromptTemplate.from_template(grader_prompt_string)
grader_model = ChatOpenAI(model='gpt-3.5-turbo')
grader_chain = grader_prompt | grader_model | parser

#feedback = grader_chain.invoke({"question": question, "student_answer": student_answer, "learning_standards": learning_standards})
#print(feedback)

feedback = '''
Key Ideas and Details: Good
-The student is able to explain the main idea of the text, which is the crucial role of the pitcher in a baseball game.
-They also provide details about the pitcher's ability to strike out batters and the impact of their pitching on the team's success.

Craft and Structure: Needs Improvement
-The student does not demonstrate an understanding of the meaning of words or phrases relevant to the subject area.
-They also do not describe the overall structure of ideas, concepts, or information in the text.

Integration of Knowledge and Ideas: Good
-The student is able to interpret the information presented in the text and explain how it contributes to an understanding of the pitcher's role in the game.

Overall, the student's answer demonstrates a good understanding of the key ideas and details and the integration of knowledge and ideas. However, they need improvement in analyzing the craft and structure of the text. They should focus on understanding the meaning of relevant words or phrases and describing the overall structure of the text in future responses.
'''

example = '''
Grader Model Feedback:
---
Key Ideas and Details: Good
-The student is able to explain the main idea of the text, which is the crucial role of the pitcher in a baseball game.
-They also provide details about the pitcher's ability to strike out batters and the impact of their pitching on the team's success.

Craft and Structure: Needs Improvement
-The student does not demonstrate an understanding of the meaning of words or phrases relevant to the subject area.
-They also do not describe the overall structure of ideas, concepts, or information in the text.

Integration of Knowledge and Ideas: Good
-The student is able to interpret the information presented in the text and explain how it contributes to an understanding of the pitcher's role in the game.

Overall, the student's answer demonstrates a good understanding of the key ideas and details and the integration of knowledge and ideas. However, they need improvement in analyzing the craft and structure of the text. They should focus on understanding the meaning of relevant words or phrases and describing the overall structure of the text in future responses.
---

Corrected Feedback:
---
Key Ideas and Details: Great
-The student is able to explain the main idea of the text, which is the crucial role of the pitcher in a baseball game.
-They also provide details about the pitcher's ability to strike out batters and the impact of their pitching on the team's success.

Craft and Structure: Good
-The student demonstrates an understanding of the meaning of words or phrases relevant to the subject area.

Integration of Knowledge and Ideas: Good
-The student is able to interpret the information presented in the text and explain how it contributes to an understanding of the pitcher's role in the game.

Overall, the student's answer demonstrates a good understanding of the key ideas and details and the integration of knowledge and ideas. 
---
'''

qa_prompt_string= f'''
Assess feedback about a student's answer to a free response question that is generated by a grading model. \
Provide your agreement with the model's feedback (yes or no), and your recommended feedback. \
Base your feedback on the question, answer, learning standards, and example below.

Question:
---
{question}
---
Student's Answer:
---
{student_answer}
---
Learning Standards:
---
{learning_standards}
---
The Grading Model's Feedback:
---
{feedback}
---
Your Agreement with the grading model
-Yes if you agree with the grading model's rating of all categories within its feedback
-No if you don't
---
Your Feedback:
---
Your feedback based on the learning standards and example below
---

Example:
---
This is an example of grading model feedback and corrected feedback.  Align your feedback with the corrected feedback.
{example}
---
'''

qa_prompt = ChatPromptTemplate.from_template(qa_prompt_string)
qa_model = ChatOpenAI(model='gpt-3.5-turbo')
qa_chain = qa_prompt | qa_model | parser

qa_results = qa_chain.invoke({"question": question, "student_answer": student_answer, "learning_standards": learning_standards, "feedback": feedback, "example": example})
print(qa_results)

qa_results = '''
Your Agreement with the grading model: No

Your Feedback:

Key Ideas and Details: Great
-The student is able to explain the main idea of the text, which is the crucial role of the pitcher in a baseball game.
-They also provide details about the pitcher's ability to strike out batters and the impact of their pitching on the team's success.

Craft and Structure: Good
-The student demonstrates an understanding of the meaning of words or phrases relevant to the subject area.

Integration of Knowledge and Ideas: Good
-The student is able to interpret the information presented in the text and explain how it contributes to an understanding of the pitcher's role in the game.

Overall, the student's answer demonstrates a good understanding of the key ideas and details and the integration of knowledge and ideas.'''
