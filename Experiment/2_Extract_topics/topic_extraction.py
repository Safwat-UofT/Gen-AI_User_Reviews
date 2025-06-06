import os
import pandas as pd
from openai import OpenAI


# File paths stage 3
# categories file contains a list of app categories
categories_file = "categories.csv"
# the directory where the large samples are located, files are in the name "category_name"_sample.csv
samples_dir = "/topic_extraction_samples"
# the name of the file where we will save the top 10 topics for each app category
output_file = "top_10_topics.csv"

# Load categories
categories_df = pd.read_csv(categories_file)

# Define prompts 1 for P2: 0 shot, 2 for P3: 3 shot, and 3 for P4: 5 shot
prompts = {
    1: "",
    2: """
    Below are 3 examples of user reviews with the corresponding high-level topics.
    Review 1: I do like the fact that it does respond fast to what you ask. It is amazing.
    Topic for review 1: AI Performance. Reviews talk about how well the AI performs in the app in terms of accuracy, reliability, speed, precision, and adaptability. 
    Review 2: I have installed the app and it has started crashing. I thought this app is good for editing photos. but it is rubbish. for this I am giving one stars 
    Topic for review 2: Technical Difficulties. Reviews report technical issues they encounter when using the application, this includes crashes and bugs.
    Review 3: Initially. it was working pretty well & having a lot of support. but for the last 1 month. it's not working at all. Its totally become garbage
    Topic for review 3: Updates and evolution. Reviews discuss the app performance over time and how well maintained it is. Some reviews report disappointment when the app stops performing as well as it used to. 

    Reviews: 
    """,
    3: """
    Below are 5 examples of user reviews with the corresponding high-level topics.
    Review 1: I do like the fact that it does respond fast to what you ask. It is amazing.
    Topic for review 1: AI Performance. Reviews talk about how well the AI performs in the app in terms of accuracy, reliability, speed, precision, and adaptability. 
    Review 2: I have installed the app and it has started crashing. I thought this app is good for editing photos. but it is rubbish. for this I am giving one stars 
    Topic for review 2: Technical Difficulties. Reviews report technical issues they encounter when using the application, this includes crashes and bugs.
    Review 3: Initially. it was working pretty well & having a lot of support. but for the last 1 month. it's not working at all. Its totally become garbage
    Topic for review 3: Updates and evolution. Reviews discuss the app performance over time and how well maintained it is. Some reviews report disappointment when the app stops performing as well as it used to. 
    Review 4: Despite odd occurrences and etc I love what gets generated most of the time
    Topic for review 4: Quality of Generated Content. Reviews discuss how good, bad, surprising, or fun the generated content came out. 
    Review 5: Wonderful app very simple interface easy to use and apt. Highly recommended.
    Topic for review 5: User Interface. Reviews talk about the user interface of the app, how intuitive, easy to use, and/or aesthetically pleasing it is.

    Reviews:
    """,
}

# Choose a prompt we switch the prompt, 1 for P2: 0 shot, 2 for P3: 3 shot, and 3 for P4: 5 shot
selected_prompt = prompts[3]
    

# Prepare the output
output_data = []
client = OpenAI()

# Process each category
for _, row in categories_df.iterrows():
    category_name = row['playstore_category']
    sample_file = os.path.join(samples_dir, f"{category_name}_sample.csv")

    if not os.path.exists(sample_file):
        print(f"Sample file for category '{category_name}' not found. Skipping.")
        continue

    # Load the random sample file
    sample_df = pd.read_csv(sample_file)

    # Extract reviews content
    reviews_content = sample_df['content'].dropna().tolist()
    reviews_string = '\n'.join(f'"{review}"' for review in reviews_content)

    prompt_opener=f"Below are user reviews of {category_name} Generative AI applications. Each review is enclosed by quotations and separated by newline escape. Provide the list of the top 10 distinct, high level topics presented in all the reviews with a brief meaning of each topic. A high level topic describes the features, functionalities, and utility of the app.\n"
    
    # Prepare OpenAI API request
    message = f"{prompt_opener}{selected_prompt}\n\n{reviews_string}"

    try:
        # Send request to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI specialized in analyzing customer reviews."},
                {"role": "user", "content": message}
            ],
            temperature=0
        )

        # Extract the response content
        api_output = response.choices[0].message.content

        # Append to output data
        output_data.append({"category": category_name, "prompt_output": api_output})
        # Save the output to a CSV file
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"Error processing category '{category_name}': {e}")

# Save the output to a CSV file
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_file, index=False)

print(f"Output saved to {output_file}")
