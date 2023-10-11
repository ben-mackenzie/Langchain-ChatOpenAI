import os
import streamlit as st
from OpenAI_API_KEY import openai_api_key

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

os.environ['OPENAI_API_KEY'] = openai_api_key

st.title('FRQ Generator')

parser = StrOutputParser()

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

student_interest_topic = st.text_input('Enter topic to generate FRQs')

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

if student_interest_topic:
    response = generator_chain.invoke({"student_interest_topic": student_interest_topic, 'number_of_frqs': number_of_frqs})
    st.write('''INSTRUCTIONS:
             Read the following passages and questions carefully and think about the answer before writing your response. \
             In writing your responses, be sure to clearly organize your writing and express what you have learned; \
             accurately and completely answer the questions being asked; \
             support your responses with examples or details from the text; and \
             write in complete sentences using correct spelling, grammar, capitalization, and punctuation.''')
    st.write(response)
