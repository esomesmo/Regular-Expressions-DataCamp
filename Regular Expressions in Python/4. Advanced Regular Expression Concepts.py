## Try another name
# You are still working on your Twitter sentiment analysis. You analyze now some things that caught your attention. You noticed that there are email addresses inserted in some tweets. Now, you are curious to find out which is the most common name.

# You want to extract the first part of the email. E.g. if you have the email marysmith90@gmail.com, you are only interested in marysmith90.
# You need to match the entire expression. So you make sure to extract only names present in emails. Also, you are only interested in names containing upper (e.g. A,B, Z) or lowercase letters (e.g. a, d, z) and numbers.

# # The list sentiment_analysis containing the text of three tweets as well as the re module were loaded in your session. You can use print() to view it in the IPython Shell.

# Instructions
# 100 XP
# Complete the regex to match the email capturing only the name part. The name part appears before the @.
# Find all matches of the regex in each element of sentiment_analysis analysis. Assign it to the variable email_matched.
# Complete the .format() method to print the results captured in each element of sentiment_analysis analysis.

# Write a regex that matches email
regex_email = r"([a-zA-Z0-9]+)@\S+"

for tweet in sentiment_analysis:
    # Find all matches of regex in each tweet
    email_matched = re.findall(regex_email, tweet)

    # Complete the format method to print the results
    print("Lists of users found in this tweet: {}".format(email_matched))
	

## Flying home
# Your boss assigned you to a small project. They are performing an analysis of the travels people made to attend business meetings. You are given a dataset with only the email subjects for each of the people traveling.

# You learn that the text followed a pattern. Here is an example:

# Here you have your boarding pass LA4214 AER-CDB 06NOV.

# You need to extract the last part:

# The two letters indicate the airline (e.g LA),
# The 4 numbers are the flight number (e.g. 4214).
# The three letters correspond to the departure (e.g AER),
# The destination (CDB),
# The date (06NOV) of the flight.
# All letters are always uppercase.

# The variable flight containing one email subject was loaded in your session. You can use print() to view it in the IPython Shell.

# Instructions 4/4
# 25 XP
# Complete the format method with the elements contained in flight_matches. In the first line print the airline,and the flight number. In the second line, the departure and destination. In the third line, the date.

# Import re
import re

# Write regex to capture information of the flight
regex = r"([A-Z]{2})(\d{4})\s([A-Z]{3})-([A-Z]{3})\s(\d{2}[A-Z]{3})"

# Find all matches of the flight information
flight_matches = re.findall(regex, flight)
    
#Print the matches
print("Airline: {} Flight number: {}".format(flight_matches[0][0], flight_matches[0][1]))
print("Departure: {} Destination: {}".format(flight_matches[0][2], flight_matches[0][3]))
print("Date: {}".format(flight_matches[0][4]))


## Love it!
# You are still working on the Twitter sentiment analysis project. First, you want to identify positive tweets about movies and concerts.

# You plan to find all the sentences that contain the words love, like, or enjoy and capture that word. You will limit the tweets by focusing on those that contain the words movie or concert by keeping the word in another group. You will also save the movie or concert name.

# For example, if you have the sentence: I love the movie Avengers. You match and capture love. You need to match and capture movie. Afterwards, you match and capture anything until the dot.

# The list sentiment_analysis containing the text of three tweets and the re module are loaded in your session. You can use print() to view the data in the IPython Shell.

# Instructions
# 100 XP
# Complete the regular expression to capture the words love or like or enjoy. Match and capture the words movie or concert. Match and capture anything appearing until the ..
# Find all matches of the regex in each element of sentiment_analysis. Assign them to positive_matches.
# Complete the .format() method to print out the results contained in positive_matches for each element in sentiment_analysis.

# Write a regex that matches sentences with the optional words
regex_positive = r"(love|like|enjoy).+?(movie|concert)\s(.+?)\."

for tweet in sentiment_analysis:
	# Find all matches of regex in tweet
    positive_matches = re.findall(regex_positive, tweet)
    
    # Complete format to print out the results
    print("Positive comments found {}".format(positive_matches))
	
	
	
	
## Ugh! Not for me!
# After finding positive tweets, you want to do it for negative tweets. Your plan now is to find sentences that contain the words hate, dislike or disapprove. You will again save the movie or concert name. You will get the tweet containing the words movie or concert but this time, you don't plan to save the word.

# For example, if you have the sentence: I dislike the movie Avengers a lot.. You match and capture dislike. You will match but not capture the word movie. Afterwards, you match and capture anything until the dot.

# The list sentiment_analysis containing the text of three tweets as well as the re module are loaded in your session. You can use print() to view the data in the IPython Shell.

# Instructions
# 100 XP
# Complete the regular expression to capture the words hate or dislike or disapprove. Match but don't capture the words movie or concert. Match and capture anything appearing until the ..
# Find all matches of the regex in each element of sentiment_analysis. Assign them to negative_matches.
# Complete the .format() method to print out the results contained in negative_matches for each element in sentiment_analysis.	

# Write a regex that matches sentences with the optional words
regex_negative = r"(hate|dislike|disapprove).+?(?:movie|concert)\s(.+?)\."

for tweet in sentiment_analysis:
	# Find all matches of regex in tweet
    negative_matches = re.findall(regex_negative, tweet)
    
    # Complete format to print out the results
    print("Negative comments found {}".format(negative_matches))
	
## Parsing PDF files
# You now need to work on another small project you have been delaying. Your company gave you some PDF files of signed contracts. The goal of the project is to create a database with the information you parse from them. Three of these columns should correspond to the day, month, and year when the contract was signed.
# The dates appear as Signed on 05/24/2016 (05 indicating the month, 24 the day). You decide to use capturing groups to extract this information. Also, you would like to retrieve that information so you can store it separately in different variables.

# You decide to do a proof of concept.

# The variable contract containing the text of one contract and the re module are already loaded in your session. You can use print() to view the data in the IPython Shell.

# Instructions 3/3
# 30 XP
# Complete the f-string method to print out the captured groups. Use the values corresponding to each key in the dictionary.

# Write regex and scan contract to capture the dates described
regex_dates = r"Signed\son\s(\d{2})/(\d{2})/(\d{4})"
dates = re.search(regex_dates, contract)

# Assign to each key the corresponding match
signature = {
	"day": dates.group(2),
	"month": dates.group(1),
	"year": dates.group(3)
}
# Complete the format method to print-out
print("Our first contract is dated back to {data[year]}. Particularly, the day {data[day]} of the month {data[month]}.".format(data=signature))


## Close the tag, please!
# In the meantime, you are working on one of your other projects. The company is going to develop a new product. It will help developers automatically check the code they are writing. You need to write a short script for checking that every HTML tag that is open has its proper closure.

# You have an example of a string containing HTML tags:

# <title>The Data Science Company</title>

# You learn that an opening HTML tag is always at the beginning of the string. It appears inside <>. A closing tag also appears inside <>, but it is preceded by /.

# You also remember that capturing groups can be referenced using numbers, e.g \4.

# The list html_tags, containing three strings with HTML tags, and there module are loaded in your session. You can use print() to view the data in the IPython Shell.

# Instructions
# 0 XP
# Complete the regex in order to match closed HTML tags. Find if there is a match in each string of the list html_tags. Assign the result to match_tag.
# If a match is found, print the first group captured and saved in match_tag.
# If no match is found, complete the regex to match only the text inside the HTML tag. Assign it to notmatch_tag.
# Print the first group captured by the regex and save it in notmatch_tag.

for string in html_tags:
    # Complete the regex and find if it matches a closed HTML tags
    match_tag =  re.match(r"<(\w+)>.*?</\1>", string)
 
    if match_tag:
        # If it matches print the first group capture
        print("Your tag {} is closed".format(match_tag.group(1))) 
    else:
        # If it doesn't match capture only the tag 
        notmatch_tag = re.match(r"<(\w+)>", string)
        # Print the first group capture
        print("Close your {} tag!".format(notmatch_tag.group(1)))

## Reeepeated characters
# Back to your sentiment analysis! Your next task is to replace elongated words that appear in the tweets. We define an elongated word as a word that contains a repeating character twice or more times. e.g. "Awesoooome".

# Replacing those words is very important since a classifier will treat them as a different term from the source words lowering their frequency.

# To find them, you will use capturing groups and reference them back using numbers. E.g \4.

# If you want to find a match for Awesoooome. You first need to capture Awes. Then, match o and reference the same character back, and then, me.

# The list sentiment_analysis, containing the text of three tweets, and the re module are loaded in your session. You can use print() to view the data in the IPython Shell.

# Instructions
# 100 XP
# Complete the regular expression to match an elongated word as described.
# Search the elements in sentiment_analysis list to find out if they contain elongated words. Assign the result to match_elongated.
# Assign the captured group number zero to the variable elongated_word.
# Print the result contained in the variable elongated_word.

# Complete the regex to match an elongated word
regex_elongated = r"\w+(\w)\1\w*"

for tweet in sentiment_analysis:
	# Find if there is a match in each tweet 
	match_elongated = re.search(regex_elongated, tweet)
    
	if match_elongated:
		# Assign the captured group zero 
		elongated_word = match_elongated.group(0)
        
		# Complete the format method to print the word
		print("Elongated word found: {word}".format(word=elongated_word))
	else:
		print("No elongated word found") 


## Surrounding words
# Now, you want to perform some visualizations with your sentiment_analysis dataset. You are interested in the words surrounding python. You want to count how many times a specific words appears right before and after it.

# Positive lookahead (?=) makes sure that first part of the expression is followed by the lookahead expression. Positive lookbehind (?<=) returns all matches that are preceded by the specified pattern.

# The variable sentiment_analysis, containing the text of one tweet, and the re module are loaded in your session. You can use print() to view the data in the IPython Shell.

# Instructions 1/2
# 50 XP
# Get all the words that are followed by the word python in sentiment_analysis. Print out the word found.

# Positive lookahead
look_ahead = re.findall(r"\w+(?=\spython)", sentiment_analysis)

# Print out
print(look_ahead)
	
# Get all the words that are preceded by the word python or Python in sentiment_analysis. Print out the words found.

# Positive lookbehind
look_behind = re.findall(r"(?<=[Pp]ython\s)\w+", sentiment_analysis)

# Print out
print(look_behind)

## Filtering phone numbers
# Now, you need to write a script for a cell-phone searcher. It should scan a list of phone numbers and return those that meet certain characteristics.

# The phone numbers in the list have the structure:

# Optional area code: 3 numbers
# Prefix: 4 numbers
# Line number: 6 numbers
# Optional extension: 2 numbers
# E.g. 654-8764-439434-01.

# You decide to use .findall() and the non-capturing group's negative lookahead (?!) and negative lookbehind (?<!).

# The list cellphones, containing three phone numbers, and the re module are loaded in your session. You can use print() to view the data in the IPython Shell.

# Instructions 1/2
# 0 XP
# Get all phone numbers that are not preceded by the optional area code.

for phone in cellphones:
	# Get all phone numbers not preceded by area code
	number = re.findall(r"(?<!\d{3}-)\d{4}-\d{6}-\d{2}", phone)
	print(number)
	
# Get all the phone numbers that are not followed by the optional extension.

for phone in cellphones:
	# Get all phone numbers not followed by optional extension
	number = re.findall(r"\d{3}-\d{4}-\d{6}(?!-\d{2})", phone)
	print(number)