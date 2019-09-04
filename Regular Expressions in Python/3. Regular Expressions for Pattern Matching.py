## Are they bots?
# The company that you are working for asked you to perform a sentiment analysis using a dataset with tweets. First of all, you need to do some cleaning and extract some information.
While printing out some text, you realize that some tweets contain user mentions. Some of these mentions follow a very strange pattern. A few examples that you notice: @robot3!, @robot5& and @robot7#

To analyze if those users are bots, you will do a proof of concept with one tweet and extract them using the .findall() method.

You write down some helpful metacharacters to help you later:

\d: digit
\w: word character
\W: non-word character
\s: whitespace

The text of one tweet was saved in the variable sentiment_analysis. You can use print(sentiment_analysis) to view it in the IPython Shell.

Instructions
100 XP
Import the re module.
Write a regex that matches the user mentions that follows the pattern, e.g. @robot3!.
Find all the matches of the pattern in the sentiment_analysis variable.

# Import the re module
import re

# Write the regex
regex = r"@robot\d\W"

# Find all matches of regex
print(re.findall(regex, sentiment_analysis))



## Find the numbers
# While examining the tweet text in your dataset, you detect that some tweets carry extra information. The text contains the number of retweets, user mentions, and likes of that tweet. So, you decide to extract this important information.

The information is given as in this example:

Agh...snow! User_mentions:9, likes: 5, number of retweets: 4

You also bring your list of metacharacters: \d digit, \w word character, \s whitespace.

The variable sentiment_analysis containing the text of one tweet and the re module were loaded in your session. You can use print() to view it in the IPython Shell.

Instructions 1/3
35 XP
Write a regex that matches the number of user mentions given as, for example, User_mentions:9 in sentiment_analysis.

# Write a regex to obtain user mentions
print(re.findall(r"User_mentions:\d", sentiment_analysis))

# Write a regex that matches the number of likes given as, for example, likes: 5 in sentiment_analysis.

# Write a regex to obtain number of likes
print(re.findall(r"likes:\s\d", sentiment_analysis))

# Write a regex that matches the number of retweets given as, for example, number of retweets: 4 in sentiment_analysis.

# Write a regex to obtain number of retweets
print(re.findall(r"number\sof\sretweets:\s\d", sentiment_analysis))


## Match and split
# Some of the tweets in your dataset were downloaded incorrectly. Instead of having spaces to separate words, they have strange characters. You decide to use regular expressions to handle this situation. You print some of these tweets to understand which pattern you need to match.

# You notice that the sentences are always separated by a special character, followed by a number, the word break, and after that, another special character, e.g &4break!. The words are always separated by a special character, the word new, and a normal random character, e.g #newH.

The variable sentiment_analysis containing the text of one tweet, as well as the re module were already loaded in your session. You can use print(sentiment_analysis) to view it in the IPython Shell.

Instructions 4/4
1 XP
Replace regex_words with a space in the variable sentiment_sub. Assign it to sentiment_final and print out the result.

# Write a regex to match pattern separating sentences
regex_sentence = r"\W\dbreak\W"

# Replace the regex_sentence with a space
sentiment_sub = re.sub(regex_sentence, " ", sentiment_analysis)

# Write a regex to match pattern separating words
regex_words = r"\Wnew\w"

# Replace the regex_words and print the result
sentiment_final = re.sub(regex_words, " ", sentiment_sub)
print(sentiment_final)

## Everything clean
Back to your Twitter sentiment analysis project! There are several types of strings that increase your sentiment analysis complexity. But these strings do not provide any useful sentiment. Among them, we can have links and user mentions.

In order to clean the tweets, you want to extract some examples first. You know that most of the times links start with http and do not contain any whitespace, e.g. https://www.datacamp.com. User mentions start with @ and can have letters and numbers only, e.g. @johnsmith3.

You write down some helpful quantifiers to help you: * zero or more times, + once or more, ? zero or once.

The list sentiment_analysis containing the text of three tweet are already loaded in your session. You can use print() to view the data in the IPython Shell.

Instructions
100 XP
Import the re module.
Write a regex to find all the matches of http links appearing in each tweet in sentiment_analysis. Print out the result.
Write a regex to find all the matches of user mentions appearing in each tweet in sentiment_analysis. Print out the result.

# Import re module
import re

for tweet in sentiment_analysis:
	# Write regex to match http links and print out result
	print(re.findall(r"http\S+", tweet))

	# Write regex to match user mentions and print out result
	print(re.findall(r"@\w+\d", tweet))


## Some time ago
# You are interested in knowing when the tweets were posted. After reading a little bit more, you learn that dates are provided in different ways. You decide to extract the dates using .findall() so you can normalize them afterwards to make them all look the same.

You realize that the dates are always presented in one of the following ways:

27 minutes ago

4 hours ago

23rd june 2018

1st september 2019 17:25

The list sentiment_analysis containing the text of three tweets, as well as the re module are already loaded in your session. You can use print() to view the data in the IPython Shell.

Instructions 1/3
35 XP
Complete the for-loop with a regex that finds all dates in a format similar to 27 minutes ago or 4 hours ago.


# Complete the for loop with a regex to find dates
for date in sentiment_analysis:
	print(re.findall(r"\d{1,2}\s\w+\s\w+", date))
	
# Complete the for-loop with a regex that finds all dates in a format similar to 23rd june 2018.

# Complete the for loop with a regex to find dates
for date in sentiment_analysis:
	print(re.findall(r"\d{1,2}\w+\s\w+\s\d{4}", date))

# Complete the for-loop with a regex that finds all dates in a format similar to 1st september 2019 17:25.

# Complete the for loop with a regex to find dates
for date in sentiment_analysis:
	print(re.findall(r"\d{1,2}\d+\s\d+\s\d{4}\s\d{1,2}:\d{2}", date))


## Getting tokens
# Your next step is to tokenize the text of your tweets. Tokenization is the process of breaking a string into lexical units or, in simpler terms, words. But first, you need to remove hashtags so they do not cloud your process. You realize that hashtags start with a # symbol and contain letters and numbers but never whitespace. After that, you plan to split the text at whitespace matches to get the tokens.

You bring your list of quantifiers to help you: * zero o more times, + once or more, ? zero or once, {n, m} minimum n, maximum m.

The variable sentiment_analysis containing the text of one tweet as well as the re module are already loaded in your session. You can use print(sentiment_analysis) to view it in the IPython Shell.

Instructions 3/3
30 XP
Split the text in the no_hashtag variable at every match of one or more consecutive whitespace.

# Write a regex matching the hashtag pattern
regex = r"#\w+"

# Replace the regex by an empty string
no_hashtag = re.sub(regex, "", sentiment_analysis)

# Get tokens by splitting text
print(re.split(r"\s+", no_hashtag))


## Finding files
You are not satisfied with your tweets dataset cleaning. There are still extra strings that do not provide any sentiment. Among them are strings refer to text file names.

You also find a way to detect them:

They appear at the start of the string.
They always start with a sequence of 2 or 3 upper or lowercase vowels (a e i o u).
They always finish with the txt ending.
You are not sure if you should remove them directly. So you write a script to find and store them in a separate dataset.

You write down some metacharacters to help you: ^ anchor to beginning, . any character.

The variable sentiment_analysis containing the text of two tweets as well as the re module are already loaded in your session. You can use print() to view it in the IPython Shell.

Instructions
100 XP
Write a regex that matches the pattern of the text file names, e.g. aemyfile.txt.
Find all matches of the regex in the elements of sentiment_analysis. Print out the result.
Replace all matches of the regex with an empty string "". Print out the result.

# Write a regex to match text file name
regex = r"^[aeiouAEIOU]{2,3}.+txt"

for text in sentiment_analysis:
	# Find all matches of the regex
	print(re.findall(regex, text))
    
	# Replace all matches with empty string
	print(re.sub(regex, "", text))


## Give me your email
A colleague has asked for your help! When a user signs up on the company website, they must provide a valid email address.
The company puts some rules in place to verify that the given email address is valid:

The first part can contain:
Upper A-Z and lowercase letters a-z
Numbers
Characters: !, #, %, &, *, $, .
Must have @
Domain:
Can contain any word characters
But only .com ending is allowed
The project consist of writing a script that checks if the email address follow the correct pattern. Your colleague gave you a list of email addresses as examples to test.

The list emails as well as the re module are loaded in your session. You can use print(emails) to view the emails in the IPython Shell.

Instructions
0 XP
Instructions
0 XP
Write a regular expression to match valid email addresses as described.
Match the regex to the elements contained in emails.
To print out the message indicating if it is a valid email or not, complete .format() statement.

# Write a regex to match a valid email address
regex = r"[A-Za-z0-9!#%&*\$\.]+@\w+\.com"

for example in emails:
  	# Match the regex to the string
    if re.match(regex, example):
        # Complete the format method to print out the result
      	print("The email {email_example} is a valid email".format(email_example=example))
    else:
      	print("The email {email_example} is invalid".format(email_example=example))   


## Invalid password
The second part of the website project is to write a script that validates the password entered by the user. The company also puts some rules in order to verify valid passwords:

It can contain lowercase a-z and uppercase letters A-Z
It can contain numbers
It can contain the symbols: *, #, $, %, !, &, .
It must be at least 8 characters long but not more than 20
Your colleague also gave you a list of passwords as examples to test.

The list passwords and the module re are loaded in your session. You can use print(passwords) to view them in the IPython Shell.

Instructions
100 XP
Write a regular expression to match valid passwords as described.
Scan the elements in the passwords list to find out if they are valid passwords.
To print out the message indicating if it is a valid password or not, complete .format() statement.

# Write a regex to match a valid password
regex = r"[a-zA-Z*#\$%!&\.0-9]{8,20}"

for example in passwords:
  	# Scan the strings to find a match
    if re.search(regex, example):
        # Complete the format method to print out the result
      	print("The password {pass_example} is a valid password".format(pass_example=example))
    else:
      	print("The password {pass_example} is invalid".format(pass_example=example))


## Understanding the difference
You need to keep working and cleaning your tweets dataset. You realize that there are some HTML tags present. You need to remove them but keep the inside content as they are useful for analysis.

Let's take a look at this sentence containing an HTML tag:

I want to see that <strong>amazing show</strong> again!.

You know that for getting HTML tag you need to match anything that sits inside angle brackets < >. But the biggest problem is that the closing tag has the same structure. If you match too much, you will end up removing key information. So you need to decide whether to use a greedy or a lazy quantifier.

The string is already loaded as string to your session.

Instructions
100 XP
Import the re module.
Write a regex expression to replace HTML tags with an empty string.
Print out the result.

# Import re
import re

# Write a regex to eliminate tags
string_notags = re.sub(r"<.+?>", "", string)
# Print out the result
print(string_notags)

## Greedy matching
# Next, you see that numbers still appear in the text of the tweets. So, you decide to find all of them.

# Let's imagine that you want to extract the number contained in the sentence I was born on April 24th. A lazy quantifier will make the regex return 2 and 4, because they will match as few characters as needed. However, a greedy quantifier will return the entire 24 due to its need to match as much as possible.

# The re module as well as the variable sentiment_analysis are already loaded in your session. You can use print(sentiment_analysis) to view it in the IPython Shell.

Instructions 1/2
50 XP
Use a lazy quantifier to match all numbers that appear in the variab

# Write a lazy regex expression 
numbers_found_lazy = re.findall(r"\d+?", sentiment_analysis)

# Print out the result
print(numbers_found_lazy)

# Now, use a greedy quantifier to match all numbers that appear in the variable sentiment_analysis.

# Write a greedy regex expression 
numbers_found_greedy = re.findall(r"\d+", sentiment_analysis)

# Print out the result
print(numbers_found_greedy)

## Lazy approach
# You have done some cleaning in your dataset but you are worried that there are sentences encased in parentheses that may cloud your analysis.

# Again, a greedy or a lazy quantifier may lead to different results.

# or example, if you want to extract a word starting with a and ending with e in the string I like apple pie, you may think that applying the greedy regex a.+e will return apple. However, your match will be apple pie. A way to overcome this is to make it lazy by using ? which will return apple.

# The re module and the variable sentiment_analysis are already loaded in your session.

# Instructions 1/2
# 50 XP
# Use a greedy quantifier to match text that appears within parentheses in the variable sentiment_analysis.

# Write a greedy regex expression to match 
sentences_found_greedy = re.findall(r"\(.+\)", sentiment_analysis)

# Print out the result
print(sentences_found_greedy)

# Now, use a lazy quantifier to match text that appears within parentheses in the variable sentiment_analysis.

# Write a lazy regex expression
sentences_found_lazy = re.findall(r"\(.*?\)", sentiment_analysis)

# Print out the results
print(sentences_found_lazy)
















