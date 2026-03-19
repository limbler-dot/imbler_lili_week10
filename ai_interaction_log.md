# SETUP: I told Codex to create and use a virtual environment called .venv, then activate it for windows, install streamlit / requests, add these to requirements.txt, and run the app. It experienced difficulty opening the Streamlit website so I had to do so manually. I set up the API and files.
# It created a plan to create and activate the virtual environment while installing the necessary packages. The packages were added to requirements.txt. 
# I implemented what it gave after extensive back and forth. 

# TASK 1A
# I told Codex to use st.set_page_config(), load my token without harcoding it, display special errors without crashing, and handle potential API errors in a user friendly manner.
# It did this smoothly. I added my token and monitored progress in the streamlit browser. 
# I implemented the code it gave to make my commands happen.

# TASK 1B
# I told Codex the given instructions. I had to tinker with the saving and layout of the app but was able to get it functioning.
# It was able to develop these commands with additional instruction for the scrolling and conversation history aspects.
# I implemented the code.

# TASK 1C
# I told codex the instructions.
# It created most of the aspects but did not highlight the current chat well. I gave it a color and manner to indicate the current chat.
# I used the code it gave after we developed the plan according to the instructions and with my personal touches for an intuitive UI.

# TASK 1D
# I told Codex to save chats in the chats/ directory, save these upon startup, and follow the other instructions.
# It saves the chats with these categories and has the functions it needs.
# It was able to implement the code to accurately and smoothly create the persistence needed for the chat.

# TASK 2
# I told Codex the parameters and what it must use.
# It created a plan to implement the parameters and ensured that only Streamlit methods would be used
# This code was added onto the current built code in app.py.

# Task 3
# It was told to extract traits and store them in memory.json, and to display particular categories in the sidebar.
# It saved the traits as wanted and made a useful sidebar.
# I added the code to app.py and made sure memory.json was operating.

# Throughout this process, I ran the success criteria and referenced the hints to ensure my program was working correctly.