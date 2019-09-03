## Put it in order!
# Your company is analyzing the best way to provide users with different online courses. Your job is to scrape Wikipedia pages searching for tools used in Data Science subfields. You'll store the tool and field name in a database. After a text analysis, you realize that the information is provided in a specific position of the text but sometimes the field name is given first and the tool after that, while in other cases it's the other way around.

# You decide to use positional formatting to handle these situations because it provides a way to reorder placeholders.

# The text of one article has already been saved in the variable wikipedia_article. Also, the empty list my_list is already defined. You can use print() to view the variable in the IPython Shell.

Instructions 4/4
25 XP
Complete the for-loop so that it uses the .format() method and the variables first_pos and second_pos to print out every string in my_list.

# Assign the substrings to the variables
first_pos = wikipedia_article[3:19].lower()
second_pos = wikipedia_article[21:44].lower()

# Define string with placeholders 
my_list.append("The tool {} is used in {}")

# Define string with rearranged placeholders
my_list.append("The tool {1} is used in {0}")

# Use format to print strings
for my_string in my_list:
  	print(my_string.format(first_pos, second_pos))


## Calling by its name
# You have created your database with the tools and the different Data Science subfields they are used in. The company is considering creating courses using these tools and sending personalized emails to the users recommending the different topics. They have asked you to make this process more time-efficient. To do this, you want to create a template email with a standard message changing the different tools and corresponding field name.

# First, you want to try doing this with just one example as a proof of concept. You use positional formatting and named placeholders to call the variables in a dictionary.

# The variable courses containing one tool and one field name has been saved. You can use print(courses) to view the variable in the IPython Shell.

# Create a dictionary
plan = {
  		"field": courses[0],
        "tool": courses[1]
        }

# Define string with placeholders
my_message = "If you are interested in {plan[field]}, you can take the course related to {plan[tool]}"

# Use dictionary to replace placehoders
print(my_message.format(plan=plan))


## What day is today?
# It's lunch time and you are talking with some of your colleagues. They comment that they feel that every morning someone should send them a reminder of what day it is so they can check in the calendar what their assignments are for that day.

# You want to help out and decide to write a small script that takes the date and time of the day so that every morning, a message is sent to your colleagues. You remember that you can use the module datetime along with named placeholders to achieve your goal.

The date should be expressed as April 16, 2019 and the time as 16:30.

You write down some specifiers to help you: %d(day), %B (month name), %m (month number), %Y(year), %H (hour) and %M(minutes)

You can use the IPython Shell to explore the module datetime.

Instructions
100 XP
Import the function datetime from the module datetime .
Obtain the date of today and assign it to the variable get_date.
Complete the string message by adding placeholders named today and the format specifiers to format the date as month_name day, year and time as hour:minutes.
Print the message using the .format() method and the variable get_date to replace the named placeholders.

# Import datetime 
from datetime import datetime

# Assign date to get_date
get_date = datetime.now()

# Add named placeholders with format specifiers
message = "Good morning. Today is {today:%B %d, %Y}. It's {today:%H:%M} ... time to work!"

# Format date
print(message.format(today=get_date))



## Literally formatting
# While analyzing the text from Wikipedia pages, you read that Python 3.6 introduced f-strings.

# You remember that you've created a website that displayed data science facts but it was too slow. You think that it could be due to the string formatting you used. Because f-strings are very fast and easy to use, you decide to rewrite that project.

The variables field1, field2 and field3 containing character strings as well as the numeric variables fact1, fact2, fact3 and fact4 have been saved.

If you want to explore the variables, you can use print() to view them in the IPython Shell.

Instructions 1/3
35 XP
Complete the f-string to include the variable field1 with quotes and the variable fact1 as a digit.

# Complete the f-string
print(f"Data science is considered {field1!r} in the {fact1}st century")

# Complete the f-string to include the the variable fact2 using exponential notation, and the variable field2.

# Complete the f-string
print(f"About {fact2:e} of {field2} in the world")

# Complete the f-string to include field3 together with fact3 rounded to 2 decimals, and fact4 rounded to one decimal.

# Complete the f-string
print(f"{field3} create around {fact3:.2f}% of the data but only {fact4:.1f}% is analyzed")

## Make this function
# Wow! You are excited to see how fast and easy f-strings worked. So you plan to rewrite some more of your old code.

# Now you know that f-strings allow you to evaluate expressions where they appear and include function and method calls. You decide to use them in a project where you analyze 120 tweets to check if they include links to different news. In that way, you expect the code to be cleaner and more readable.

# The variables number1, number2,string1, and list_links have already been pre-loaded.

If you want to explore the variables, you can use print() to view them in the IPython Shell.

Instructions 1/3
35 XP
Inside the f-string, include number1,number2 and the result of dividing number1 by number2 rounded to one decimal.

# Include both variables and the result of dividing them 
print(f"{number1} tweets were downloaded in {number2} minutes indicating a speed of {number1/number2:.1f} tweets per min")

# Inside the f-string, use .replace() to replace the substring https with an empty substring in string1.

# Replace the substring http by an empty string
print(f"{string1.replace('https', '')}")

# Inside the f-string, get list_links length, multiply it by 100 and divide it by 120. Round the result to two decimals.

# Divide the length of list by 120 rounded to two decimals
print(____"Only ____% of the posts contain links")



## On time
# Lastly, you want to rewrite an old real estate prediction project. At the time, you obtained historical information about house prices and used it to make a prediction on future values.

# The date was in the datetime format: datetime.datetime(1990, 3, 17) but to print it out, you format it as 3-17-1990. You also remember that you defined a dictionary for each neighborhood. Now, you believe that you can handle both type of data better with f-strings.

# Two dictionaries, east and west, both with the keys date and price, have already been loaded. You can use print() to view them in the IPython Shell.

Instructions 1/2
0 XP
Inside the f-string, access the values of the keys price and date in east dictionary. Format the date to month-day-year.

# Access values of date and price in east dictionary
print(f"The price for a house in the east neighborhood was ${east['price']} in {east['date']:%m-%d-%Y}")

# Inside the f-string, access the values of the keys price and date in west dictionary. Format the date to month-day-year.

# Access values of date and price in west dictionary
print(f"The price for a house in the west neighborhood was ${west['price']} in {west['date']:%m-%d-%Y}.")

## Preparing a report
# Once again, you scraped Wikipedia pages. This time, you searched for the description of useful tools used for text mining. Your first task is to prepare a report about different tools you found. You want to format the information contained in the dataset to be printed out as: The (tool) is a (description).

In this case, template strings are the best solution to interpolate data generated by external sources into an already created template.

For this example, the variables tool1, tool2 and tool3 contain three article titles. Each variable description1, description2 and description3 contains the corresponding article description.

If you want to explore the variables, you can use print() to view them in the IPython Shell.

Instructions 3/3
30 XP
Substitute identifiers with the correct tool and description variables in the template and print out the results.

# Import Template
from string import Template

# Create a template
wikipedia = Template("$tool is a $description")

# Substitute variables in template
print(wikipedia.substitute(tool=tool1, description=description1))
print(wikipedia.substitute(tool=tool2, description=description2))
print(wikipedia.substitute(tool=tool3, description=description3))

## Identifying prices
# After you showed your report to your boss, he came up with the idea of offering courses to the company's users on some of the tools you studied. In order to make a pilot test, you will send an email offering a course about one of the tools, randomly chosen from your dataset. You also mention that the estimated fee needs to be paid on a monthly basis.

For writing the email, you will use Template strings. You remember that you need to be careful when you use the dollar sign since it is used for identifiers in this case.

For this example, the list tools contains the corresponding tool name, fee and payment type for the product offer. If you want to explore the variable, you can use print() to view it in the IPython Shell.

Instructions
100 XP
Assign the first, second, and third element of tools to the variables our_tool, our_fee and our_pay respectively.
Complete the template string using $tool, $fee, and $pay as identifiers. Add the dollar sign before the $fee identifier and add the characters ly directly after the $pay identifier.
Substitute identifiers with the three variables you created and print out the results.

# Import template
from string import Template

# Select variables
our_tool = tools[0]
our_fee = tools[1]
our_pay = tools[2]

# Create template
course = Template("We are offering a 3-month beginner course on $tool just for $$ $fee ${pay}ly")

# Substitute identifiers with three variables
print(course.substitute(tool=our_tool, fee=our_fee, pay=our_pay))

## Playing safe
# You are in charge of a new project! Your job is to start collecting information from the company's main application users. You will make an online quiz and ask your users to voluntarily answer two questions. However, it is not mandatory for the user to answer both. You will be handling user-provided strings so you decide to use the Template method to print the input information. This allows users to double-check their answers before submitting them.

The answer of one user has been stored in the dictionary answers. You can use the print() function to view the variables in the IPython Shell.

Instructions 2/3
25 XP
Complete the template string using $answer1 and $answer2 as identifiers.

Use the method .substitute() to replace the identifiers with the values in answers in the predefined template.

# Import template
from string import Template

# Complete template string using identifiers
the_answers = Template("Check your answer 1: $answer1, and your answer 2: $answer2")

# Use substitute to replace identifiers
try:
    print(the_answers.substitute(answers))
except KeyError:
    print("Missing information")

# Use the method .safe_substitute() to replace the identifiers with the values in answers in the predefined template.

# Use safe_substitute to replace identifiers
try:
    print(the_answers.safe_substitute(answers))
except KeyError:
    print("Missing information")


