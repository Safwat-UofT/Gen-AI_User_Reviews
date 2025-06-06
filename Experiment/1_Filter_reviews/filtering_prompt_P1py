from openai import OpenAI
import csv
import json
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Increase the field size limit to the maximum allowed by the system if you are using bigger files
#csv.field_size_limit(sys.maxsize)


# Function to handle OpenAI API request with retries
def ask_gpt_with_retries(model, message, retries=3):
    # the OpenAI API key is set in the environment
    client = OpenAI()
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a research assistant who is good at thematic coding analysis."},
                    {"role": "user", "content": message}
                ],
                temperature=0
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e


# Function to format the API response
def format_response(response):
    if response.startswith("```csv\n"):
        response = response[7:-3]
    reviews = []
    for line in response.splitlines():
        line = line.split(",")
        if len(line) == 3:
            reviews.append({"review_ID": line[0], "content": line[1], "informative?": line[2]})
    return reviews


# Function to log errors and progress
def log_error(batch, time_taken, exception_log, log_file):
    with open(log_file, "a") as file:
        file.write(f"{batch},{time_taken},{exception_log}\n")


# Function to handle a single batch
def process_batch(reviews, log_file, batch):
    start_time = datetime.now()
    exception_log = ""
    examples = """
Review "pro123": "I love it"
Output: "pro123", "I love it","non-informative"
Review "pro234": "Supper photo editor"
Output: "pro234", "Supper photo editor","non-informative"
Review "pro345": "Thanks for this apps"
Output: "pro345", "Thanks for this apps”,"non-informative"
Review "pro346": "Terrible"
Output: "pro346", "Terrible”,"non-informative"
Review "pro347": "It's really a extra ordinary fantastic app."
Output: "pro347", "It's really a extra ordinary fantastic app.”,"non-informative"
Review "pro456": "The app is not working on my phone ...why is this happening after I can't pay to keep the subscription...what is this ?????"
Output: "pro456", "The app is not working on my phone ...why is this happening after I can't pay to keep the subscription...what is this ?????","informative"
Review "pro567": "Terrible, too many ads"
Output: "pro567", “Terrible, too many ads","informative"
Review "pro789": "I love it. A very good editing experience"
Output: "pro789", “I love it. A very good editing experience","informative"
Review "pro589": "This is good app for medium black people"
Output: "pro589", “This is good app for medium black people","informative"
Review "pro689": "I love this app I just wish you didn't have to pay for unlimited tries at a.i art"
Output: "pro689", “I love this app I just wish you didn't have to pay for unlimited tries at a.i art","informative"
"""
    try:
        prompt = (
            f"I will provide you with a list of reviews and I would like to filter out the short non-informative reviews that do not provide any good input."
            f"Assign 'informative' for informative reviews and 'non-informative' for non-informative reviews. "
            f"A review is informative if it talks about a feature or functionality of the app or talks about the utility of the app (how users incorporate the app in their life or how it has helped them). A review is non-informative if it is short and generic. Non-informative reviews may express generic satisfaction and recommendation without stating exactly what they like about the app and why they recommend it.\n\n"
            f"Here are some examples"
            f"{examples}\n\n"
            f"Provide the output as CSV content: review_ID, content, informative?.\n\n"
            f"These are the review_IDs with their corresponding reviews:\n\n{reviews}"
        )
        response = ask_gpt_with_retries("gpt-4o-mini", prompt)
        formatted_reviews = format_response(response)
    except Exception as e:
        exception_log = str(e)
        formatted_reviews = []
    finally:
        time_taken = (datetime.now() - start_time).total_seconds() / 60
        log_error(batch, time_taken, exception_log, log_file)
    return formatted_reviews


# Generator to yield batches of reviews from a CSV file
def generate_batches(file_path, batch_size):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        batch = []
        for row in reader:
            review = f'{row["review_id"]}: {row["content"]}\n'
            batch.append(review)
            if len(batch) == batch_size:
                yield "".join(batch)
                batch = []
        if batch:
            yield "".join(batch)


# Function to save a checkpoint
def save_checkpoint(batch_number, checkpoint_file, results, outname):
    with open(checkpoint_file, 'w') as f:
        f.write(str(batch_number))
    #save results
    with open(outname, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# Function to load a checkpoint
def load_checkpoint(checkpoint_file):
    try:
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0

# converts a JSON file to a CSV file.
# json_path (str): Path to the input JSON file.
# csv_path (str): Path to the output CSV file.
def convert_json_to_csv(json_path, csv_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        if not data:
            print(f"No data found in {json_path}. Skipping conversion.")
            return

        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"Successfully converted {json_path} to {csv_path}")
    except Exception as e:
        print(f"Error converting {json_path} to CSV: {e}")


# Main function to process reviews and assign topics
def determine_informative_reviews(cat_file_name, outname, log_file, batch_size=100):
    # Initialize log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file = f"{log_file}_{timestamp}.csv"
    with open(log_file, "a") as file:
        file.write("batch,time,exception\n")

    # Load checkpoint
    checkpoint_file = f"{outname}_checkpoint.txt"
    start_batch = load_checkpoint(checkpoint_file)

    results = []
    batch_number = 0

    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
        futures = []
        for batch_reviews in generate_batches(cat_file_name, batch_size):
            batch_number += 1
            if batch_number < start_batch:
                continue
            futures.append(
                executor.submit(process_batch, batch_reviews, log_file, batch_number)
            )

        for future in futures:
            results.extend(future.result())
            save_checkpoint(batch_number, checkpoint_file, results, outname)  # Save progress iteratively

    # Save results to output files
    with open(outname.replace(".csv", ".json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Convert JSON to CSV if needed 
    convert_json_to_csv(outname.replace(".csv", ".json"), outname)


# Function to remove duplicate headers from the CSV file
def remove_duplicate_headers(csv_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Normalize header by stripping spaces and storing as lowercase
    header = lines[0].strip().lower().replace(", ", ",")
    unique_lines = [lines[0]]  # Always keep the original header

    for line in lines[1:]:
        # Compare normalized line with normalized header
        if line.strip().lower().replace(", ", ",") != header:
            unique_lines.append(line)

    # Write back the cleaned content
    with open(csv_file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(unique_lines)


# File paths
# Process reviews for multiple categories
categories = ["Photography", "Productivity", "Art_and_Design"]

for category in categories:
    cat_file_name = f'/1_Filter_reviews/{category}_informative_labeled.csv'
    outname = f'/1_Filter_reviews/results/{category}_filtered.csv'
    log_file = f'/1_Filter_reviews/logs/{category}_sample'

    # Run the function
    determine_informative_reviews(cat_file_name, outname, log_file)
    remove_duplicate_headers(outname)

