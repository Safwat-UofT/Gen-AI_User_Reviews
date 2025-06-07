from openai import OpenAI
import csv
import json
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json_to_csv

# Increase the field size limit to the maximum allowed by the system
csv.field_size_limit(sys.maxsize)


# Function to handle OpenAI API request with retries
# We set up OpenAI's API key in the environment
def ask_gpt_with_retries(model, message, retries=3):
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
            reviews.append({"review_ID": line[0], "content": line[1], "topic": line[2]})
    return reviews


# Function to log errors and progress
def log_error(batch, time_taken, exception_log, log_file):
    with open(log_file, "a") as file:
        file.write(f"{batch},{time_taken},{exception_log}\n")


# Function to handle a single batch
def process_batch(topics, examples, reviews, log_file, batch):
    start_time = datetime.now()
    exception_log = ""
    try:
        prompt = (
            f"Assign only one topic from the list below to each user review. "
            f"Assign the topic 'Other' if you can’t find a corresponding topic from the list. "
            f"Never assign a topic that is not from the list.\n\n"
            f"List of 10 high-level topics and their descriptions:\n{topics}\n\n"
            f"Examples: \n{examples}\n\n"
            f"Provide the output as csv content: review_ID, content, topic. Review_ID will be the provided ID for the review, the content will be the unchanged provided review, and the topic will be the topic you assigned."
            f"\nThese are the review_ID’s with their corresponding review in quotation. The reviews are separated by a new line escape. Please assign a topic to all the following reviews:\n\n{reviews}"
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
            review = f'{row["unique_id"]}: {row["content"]}\n'
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


# Main function to process reviews and assign topics
def assign_topics_to_category(cat_file_name, outname, topics, examples, log_file, batch_size=100):
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
                executor.submit(process_batch, topics, examples, batch_reviews, log_file, batch_number)
            )

        for future in futures:
            results.extend(future.result())
            save_checkpoint(batch_number, checkpoint_file, results, outname)  # Save progress iteratively

    # Save results to output files
    with open(outname, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Convert JSON to CSV if needed (assuming a utility function json_to_csv exists)
    json_to_csv.json_to_csv(outname, outname)


examples = {"Photography":'''
Review "pho123": "Did what needed to be done in style"
Output: "pho123", "Did what needed to be done in style","Other"
Review "pho234": "When i was editing it will automatically get closed and ask to report on every time for every photo. Please let me know what happened."
Output: "pho234", "When i was editing it will automatically get closed and ask to report on every time for every photo. Please let me know what happened.","Technical Issues"
Review "pho345": "Works exactly as designed and expected. Ads are only minimally obtrusive"
Output: "pho345", "Works exactly as designed and expected. Ads are only minimally obtrusive","Ad Experience"
Review "pho456": "Simply amazing! Easy to use. ;)"
Output: "pho456", "Simply amazing! Easy to use. ;)","User Experience"
Review "pho567": "Generates perfect picture with precise body size. Best Ai photo editor I have encountered"
Output: "pho567", "Generates perfect picture with precise body size. Best Ai photo editor I have encountered","AI Performance"
''',
"Productivity":"""
Review "pro123": "A very useful app helps me more than I expected. thank you very much"
Output: "pro123", "A very useful app helps me more than I expected. thank you very much","Other"
Review "pro234": "best so in the world so far it's been the best I've ever used .it isn't giving any detestable ads and unessesary payment"
Output: "pro234", "best so in the world so far it's been the best I've ever used .it isn't giving any detestable ads and unessesary payment","Pricing and Subscription Model"
Review "pro345": "Really like this app! It generates great messages. Still trying it out but thus far I'm greatly impressed"
Output: "pro345", "Really like this app! It generates great messages. Still trying it out but thus far I'm greatly impressed","Quality of Generated Content"
Review "pro456": "This is the best app to learn anything"
Output: "pro456", "This is the best app to learn anything","Educational Utility"
Review "pro567": "Wonderful app very simple interface easy to use and apt. Highly recommended."
Output: "pro567", “Wonderful app very simple interface easy to use and apt. Highly recommended.","User Interface (UI)"
""",
"Art & Design":"""
Review "art123": "Fantastic app wow this is a next level app."
Output: "art123", "Fantastic app wow this is a next level app.","Other"
Review "art234": "love how user friendly this is! It allows you to create amazing content"
Output: "art234", "love how user friendly this is! It allows you to create amazing content","User Interface"
Review "art345": "Fix your music issues and I'll give you a five star review."
Output: "art345", "Fix your music issues and I'll give you a five star review.","Technical Difficulties"
Review "art456": "the app is good for design"
Output: "art456", "the app is good for design","Practical Use Cases"
Review "art567": "sometimes the AI act dumb"
Output: "art567", “sometimes the AI act dumb","AI Performance"
""",
"Education":"""
Review "edu123": "This is so helpful"
Output: "edu123", "This is so helpful","Other"
Review "edu234": "Whenever I ask any question 80 percent chances are of it being wrong. It analyzes correctly but gives answers wrong. Even being AI powered it is not functioning correctly."
Output: "edu234", "Whenever I ask any question 80 percent chances are of it being wrong. It analyzes correctly but gives answers wrong. Even being AI powered it is not functioning correctly.","AI Performance"
Review "edu345": "I love this app..it helps me a lot about my assignments and in essays"
Output: "edu345", "I love this app..it helps me a lot about my assignments and in essays","Learning Support"
Review "edu456": "Why do i need to pay to get the answers??? It used to be free i do not recommend"
Output: "edu456", "Why do i need to pay to get the answers??? It used to be free i do not recommend","Subscription Model"
Review "edu567": "so helpful for my math and science five stars it is!!"
Output: "edu567", “so helpful for my math and science five stars it is!!","Content Variety"
""",
"Tools":"""
Review "too123": "This is so helpful"
Output: "too123", "This is so helpful","Other"
Review "too234": "Useless app it's slow and keeps on crushing. Just a waste of time."
Output: "too234", "Useless app it's slow and keeps on crushing. Just a waste of time.","Technical Difficulties"
Review "too345": "The UI is cluttered and unbecoming of a modern app which strives to be a competitor of Google."
Output: "too345", "The UI is cluttered and unbecoming of a modern app which strives to be a competitor of Google.","User Interface (UI)"
Review "too456": "Doesn't reply to my questions usually and keeps telling me to start over again!! Very disappointed and dissatisfied."
Output: "too456", "Doesn't reply to my questions usually and keeps telling me to start over again!! Very disappointed and dissatisfied.","AI Performance"
Review "too567": "The Copilot feature makes other chatgpt apps seem underbaked and redundant."
Output: "too567", “The Copilot feature makes other chatgpt apps seem underbaked and redundant.","Comparative Analysis"
""",
"Books & Reference":"""
Review "boo123": "This is an excellent application"
Output: "boo123", "This is an excellent application","Other"
Review "boo234": "It only allowed me to generate 2 stories before blocking me behind a pay wall."
Output: "boo234", "It only allowed me to generate 2 stories before blocking me behind a pay wall.","Limitations of Free Version"
Review "boo345": "A great way to break through writer's block and get the start of a story that you can edit to make it worth reading."
Output: "boo345", "A great way to break through writer's block and get the start of a story that you can edit to make it worth reading.","Creativity and Idea Generation"
Review "boo456": "Cant afford this. This is great app. I wish there was a way I could watch ad to generate stories"
Output: "boo456", "Cant afford this. This is great app. I wish there was a way I could watch ad to generate stories","Subscription Model"
Review "boo567": "The AI completely wrote the opposite of what I want and ruined the story."
Output: "boo567", “The AI completely wrote the opposite of what I want and ruined the story.","AI Performance"
"""}

topics = {
    "Photography":"""
1. **Ad Experience**  
   Reviews discuss the prevalence of ads within the app, noting how ads appear frequently during usage, leading to frustrations and interruptions in the editing process. Many users express that the ads are too long, repetitive, or obstructive.

2. **AI Performance**  
   Users comment on the app's AI capabilities, specifically regarding photo enhancement, AI-generated art, and the quality of the results. Some reviews mention inconsistencies in AI-generated outcomes, often comparing current performance unfavorably to older version capabilities.

3. **Pricing and Subscription**  
   Reviews consistently highlight dissatisfaction with the app’s pricing model, citing a lack of free features. Users voice frustration around subscription fees, the perceived value of paid features vs. functionality reasons to subscribe.

4. **User Experience**  
   Feedback revolves around the overall experience of using the app, including its intuitiveness and ease of navigation. Some users appreciate user-friendly interfaces, while others recount difficulties in accessing features, such as bugs or poor responsiveness.

5. **Feature Set and Updates**  
   Reviews discuss the range of available features in the app and how recent updates have changed previously available functionalities. Users express disappointment due to key features being removed or altered, disrupting their editing workflow.

6. **Technical Issues**  
   This topic encompasses bugs, crashes, and errors within the app, including connection issues, the app freezing, or not saving images properly. Technical reliability is a common concern among users.

7. **Quality of Editing Tools**  
   Users provide feedback on the effectiveness of editing tools, mentioning the quality of filters and results they can achieve. Some find the tools adequate for standard edits, while others expect more professional-grade options.

8. **Comparison to Other Apps**  
   Many reviews compare this app to competitors, noting either advantages or disadvantages. Users often suggest alternative apps they believe offer better value or functionality.

9. **Customer Support and Refund Issues**  
   Some users mention challenges in contacting customer support or resolving subscription-related issues. Complaints revolve around perceived lack of response or assistance regarding billing problems.

10. **Creativity and Personalization**  
   Despite frustrations, many users appreciate the creative opportunities the app provides for personalizing and enhancing images. They enjoy using features to express creativity, such as filters, stickers, and textures.
""",
"Productivity":"""
1. **AI Performance**: Reviews assess the effectiveness of the AI in answering questions accurately and quickly. Variances in response times and reliability are key concerns, with many users appreciating accurate and fast answers to their queries.

2. **Technical Difficulties**: Users report issues with app functionality, including crashes, sign-in problems, and bugs. Many express frustration with frequent errors or inability to access features they previously enjoyed.

3. **Updates and Evolution**: Feedback reflects on how the app has changed over time, with some noting improvements while others complain about degraded service after updates. Users expect ongoing support and updates to enhance functionality.

4. **Quality of Generated Content**: Discussions about the actual content created by the AI, including the relevance, accuracy, and usefulness of outputs. Users often express disappointment if the AI generates incorrect or nonsensical information.

5. **User Interface (UI)**: Reviews comment on the design and usability of the app. Features such as ease of navigation, layout, readability, and aesthetics contribute to overall user satisfaction or frustration.

6. **Pricing and Subscription Model**: Users discuss the cost associated with the app, some considering it too expensive or expressing dissatisfaction regarding the transition from a free to a paid model. There's a desire for more transparency and value in subscription plans.

7. **Educational Utility**: A significant number of reviews highlight the app's potential as a learning tool. Users, particularly students, mention how the app aids in their academic pursuits, providing explanations and assisting with complex topics.

8. **Voice Features and Interaction**: The ability of the app to understand and engage through voice commands is frequently mentioned. Users appreciate features like voice-to-text and voice interaction but often note inconsistencies in performance.

9. **Content Policy and Limitations**: Some users express frustration regarding the app's response limitations, especially concerning sensitive topics or content policies that restrict discussions. This includes dissatisfaction with how some questions trigger vague or irrelevant responses.

10. **Personalization and Memory**: Users express a desire for AI to remember previous interactions or to tailor responses based on past questions. There’s also a sentiment that enhanced personalization could improve the user experience.
""",
"Art & Design":"""
1. **AI Performance**: Users comment on the effectiveness of the AI in generating desired images or artwork, encompassing factors like accuracy, creativity, and responsiveness to prompts.

2. **Technical Difficulties**: Many reviews detail issues such as app crashes, slow performance, bugs, and connectivity problems that hinder the app's usability.

3. **Subscription Model and Pricing**: Frequent references to the cost associated with using the app, including complaints about hidden fees, subscription frustrations, and limitations on free usage.

4. **Quality of Generated Content**: Reviews often express opinions on the overall quality of images produced by the app, describing them as either stunning or disappointing, particularly when frustrating discrepancies occur between user expectations and actual results.

5. **User Interface**: Feedback addresses how intuitive, user-friendly, or aesthetically pleasing the app is, highlighting how it impacts user experience and productivity.

6. **Content Restrictions and Censorship**: Several users express frustration with perceived limitations in content creation, often related to community standards, resulting in blocked prompts or undesirable generated content.

7. **Advertising Frequency and Intrusiveness**: Users note the prevalence of ads within the app and how frequently they interrupt the creative process, arguing that it compromised usability, especially with long or unskippable ads.

8. **Updates and App Evolution**: Reviews highlight how app changes over time affect performance and features, often referencing nostalgia for previous functionalities before updates made significant changes.

9. **User Support and Community Interaction**: Experiences regarding customer service responsiveness and issues related to account management or troubleshooting are mentioned, with varying levels of satisfaction.

10. **Practical Use Cases**: Users often share their applications of the app, such as for personal projects, business needs, educational purposes, or content creation for social media, illustrating the versatility and utility that some users find valuable.
""",
"Education":"""
1. **AI Performance**: This topic encompasses how well the AI executes tasks, including the speed, accuracy, and reliability of the responses to users' queries.

2. **Quality of Generated Content**: This refers to the usefulness and coherence of the responses provided by the AI, focusing on whether the information meets the users' educational needs effectively.

3. **User Experience**: This addresses the overall ease of use of the app, including the user interface, navigation, and interaction features that enhance the learning experience.

4. **Technical Difficulties**: This covers issues such as bugs, crashes, and application performance problems that users experience, which can hinder their experience and effectiveness of the app.

5. **Subscription Model**: This entails discussions around the app’s pricing structure, including user frustrations about paywalls, ticket systems, and complaints about needing subscriptions to access certain features.

6. **Learning Support**: This focuses on the educational value provided by the app, emphasizing whether it genuinely assists users in understanding their topics or solving difficult problems.

7. **Content Variety**: This talks about the app's ability to handle different subjects beyond just mathematics, such as science or language arts, which enhances its utilization for diverse academic needs.

8. **Updates and Evolution**: This reflects users' opinions on changes made to the app over time, including any improvements or regressions in quality and features post-updates.

9. **Accessibility and Inclusivity**: This highlights whether the app is user-friendly for all demographics, including students from diverse educational backgrounds and varying financial capacities.

10. **Comparison with Other Apps**: This topic includes reviews where users analyze and compare the app against other educational tools and solutions, such as the strengths and weaknesses relative to alternatives.
""","Tools":"""
1. **AI Performance**: Reviews mention the accuracy, speed, and effectiveness of the AI capabilities. Users expect quick responses and precise information retrieval, and their feedback often refers to how the AI meets or fails these expectations. 

2. **Technical Difficulties**: Users frequently report problems such as crashes, lag, and issues related to app stability, including inability to load images or webpages, making the experience frustrating and disappointing.

3. **User Interface (UI)**: Many reviews highlight the app's design, usability, and navigation. An intuitive and aesthetically pleasing UI is often appreciated, while cluttered and confusing layouts are criticized.

4. **Quality of Generated Content**: Reviews discuss the quality of content produced by the AI, such as text generation and image creation. Users often express satisfaction with creative outputs, but may also highlight inaccuracies or irrelevance in the results.

5. **Updates and Evolution**: Some users comment on how the app has changed over time, either improving with new features or declining in quality and performance following updates.

6. **Rewards System**: A significant theme revolves around the Microsoft Rewards program, where users comment on earning points through searches and their ability to redeem these points for rewards. Reports of inconsistencies and issues in this system are common.

7. **Accessibility**: Reviews mention geographic restrictions that prevent certain features from being accessible to users in specific regions, leading to frustration, particularly in regions like India.

8. **Limitations of Features**: Users often express their frustrations about the limitations on usage (e.g., a cap on daily queries in chat), which hinders the overall effectiveness of the app's AI capabilities.

9. **Comparative Analysis**: Many reviews compare Bing with other search engines or AI tools, assessing its strengths and weaknesses relative to competitors like Google and ChatGPT, often highlighting a perceived inferiority in search accuracy or response quality.

10. **User Experience and Engagement**: This topic encompasses how enjoyable and engaging users find the app. Positive experiences often lead to high ratings, while negative, frustrating experiences lead to low ratings or uninstallation.
""","Books & Reference":"""
1. **Subscription Model**: Reviews frequently highlight the requirement for a paid subscription to access key features of the app, often expressing frustration over the limitations placed on free users and feeling deceived by marketing that implies free access.

2. **AI Performance**: Many users comment on the AI's ability to generate stories, including its coherence, creativity, and ability to meet user prompts. Performance can vary, with some praising the output, while others criticize its repetitive or shallow nature.

3. **User Interface (UI)**: This topic encompasses the design and usability of the app interface. Some reviews mention it as intuitive and simple, while others find it lacking in features and frustrating to navigate.

4. **Limitations of Free Version**: Numerous reviews note the constraints of the free version, including the number of stories that can be generated without payment. Users express dissatisfaction with only having access to a couple of stories before being prompted to upgrade.

5. **Technical Issues**: Users report various bugs or errors, including difficulties with story generation and application crashes. These problems affect the overall usability and experience of the app.

6. **Quality of Generated Content**: Content quality is a recurring theme, with some users appreciating the ability to produce good stories while others criticize the generated content for being generic, unoriginal, or poorly developed.

7. **Customer Support**: Several reviews mention interactions (or lack thereof) with customer support, citing a poor response rate or lack of assistance regarding billing issues, app functionality, or account management.

8. **Creativity and Idea Generation**: Users commonly point out the app's utility for sparking creativity, helping to overcome writer’s block, or providing inspiration for their own story developments.

9. **Feature Set and Functionality**: Comments regarding the overall features of the app include desires for additional tools such as more genres, editing capabilities, or the ability to visualize characters and settings. Users often express a need for more customization options.

10. **Value for Money**: Reviews frequently discuss whether the subscription cost aligns with the value provided by the app. Many express that the level of service or quality of generated stories does not justify the expense, leading to dissatisfaction among paying users.
"""
}

#sample Photography Productivity Art & Design
category="Photography"
examples =examples[category]
topics = topics[category]

#cat_file_name is the file that contains all the reviews from one category
cat_file_name='/input_path/'+category+'.csv'
#outname is the name of the file that will contain the results
outname='/output_path/'+category+'.csv'
#log_file keeps track of the logs
log_file='/log_path/'+category
# Run the function
assign_topics_to_category(cat_file_name, outname, topics, examples, log_file)
