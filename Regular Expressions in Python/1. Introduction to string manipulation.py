##  First day!
# Congratulations! It's your first day as a data scientist in the company! Your first project is # to build a model for predicting if a movie will get a positive or negative review.
# You need to start exploring your dataset. In order to create a function that will scan each movie review, you want to know how many characters every string has and print the result out together with a statement that indicate what the number refers to. To test if your function works correctly, you are going to start by analyzing only one example.

# The text of one movie review has been already saved in the variable movie. You can use print(movie) to view the variable in the IPython Shell.

# Find characters in movie variable
length_string = len(movie)

# Convert to string
to_string = str(length_string)

# Predefined variable
statement = "Number of characters in this review:"

# Concatenate strings and print result
print(statement + " " + to_string)

## Artificial reviews
# While checking out the movie reviews in your dataset, you realize that some of them show an atypical pattern. Since you should only include true reviews in your analysis, you decide to extract the suspicious ones that follow this pattern. You want to see if it is possible to artificially create reviews by using the first and last part of one example review and changing a keyword in the middle.

# Select the first 32 characters of movie1
first_part = movie1[:32]

# Select from 43rd character to the end of movie1
last_part = movie1[42:]

# Select from 33rd to the 42nd character
middle_part = movie2[32:42]

# Print concatenation and movie2 variable
print(first_part+last_part+middle_part) 
print(movie2)

## Palindromes
# Next, you are committed to find any peculiarity in the words included in the movie review dataset. A palindrome is a sequence of characters which can be read the same backward as forward, for example: Madam or No lemon, no melon. You realize that there are some funny movie names that can have this characteristic. You want to make a list of all movie titles that are funny palindromes but you will start by analyzing one example.

# In python, you can also specify steps by using a third index. If you don't specify the first or second index and the third one is negative, it will return the characters jumping and backwards.

# The text of a movie review for one example has been already saved in the variable movie. You can use print(movie) to view the variable in the IPython Shell.

Instructions
100 XP
Extract the substring from the 12th to the 30th character from the variable movie which corresponds to the movie title. Store it in the variable movie_title.
Get the palindrome by reversing the string contained in movie_title.
Complete the code to print out the movie_title if it is a palindrome.

# Get the word
movie_title = movie[11:30]

# Obtain the palindrome
palindrome = movie_title[::-1]

# Print the word if it's a palindrome
if movie_title == palindrome:
	print(movie_title)


## Normalizing reviews
# It's time to extract some important words present in your movie review dataset. First, you need to normalize them and then, count their frequency. Part of the normalization implies converting all the words to lowercase, removing special characters and extracting the root of a word so you count the variants as one.

# So imagine you have the following reviews: The movie surprises me very much and Marvel movies always surprise their audience. If you count the word frequency, you will count surprises one time and surprise one time. However, the verb surprise appears in both and its frequency should be two.

# The text of a movie review for only one example has been already saved in the variable movie. You can use print(movie) to view the variable in the IPython Shell.

Instructions 4/4
25 XP
To get the root of the second word contained in movie_split, select all the characters except the last one.

# Convert to lowercase and print the result
movie_lower = movie.lower()
print(movie_lower)

# Remove whitespaces and print the result
movie_no_space = movie_lower.strip("$")
print(movie_no_space)

# Split the string into substrings and print the result
movie_split = movie_no_space.split()
print(movie_split)

# Select root word and print the resul
# Remember that if you don't specify any arguments in strip() and split(), it will always remove or split whitespaces
word_root = movie_split[1][:-1]
print(word_root)

## Time to join!
# While normalizing your text, you noticed that one review had a particular structure. This review ends with the HTML tag <\i> and it has a lot of commas in different places of the sentence. You decide to remove the tag from the end and use the strategy of splitting the string and joining it back again without the commas.

# The text of a movie review has been already saved in the variable movie. You can use print(movie) to view the variable in the IPython Shell.

Instructions 1/3
35 XP
Remove tag <\i> from the end of the string. Print the results.

# Remove tags happening at the end and print results
movie_tag = movie.rstrip("<\i>")
print(movie_tag)

# Split the string using commas and print results
movie_no_comma = movie_tag.split(",")
print(movie_no_comma)

# Join back together and print results
# Remember that if you specify an argument in strip() and split(), it will use that character to remove or split.
movie_join = " ".join(movie_no_comma)
print(movie_join)


## Split lines or split the line?
# You are about to leave work when a colleague asks you to use your string manipulation skills to help him. You need to read strings from a file in a way that if the file contains strings on different lines, they are stored as separate elements. He also wants you to break the strings into pieces if you see that they contain commas.

# The text of the file has been already saved in the variable file. You can use print(file) to view the variable in the IPython Shell.

Instructions
100 XP
Split the string file into many substrings at line boundaries.
Print out the resulting variable file_split.
Complete the for-loop to split the strings into many substrings using commas as a separator element.


# Split string at line boundaries
file_split = file.split(sep="\n")

# Print file_split
print(file_split)

# Complete for-loop to split by commas
for substring in file_split:
    substring_split = substring.split(',')
    print(substring_split)

## Finding a substring
# It's a new day at work and you need to continue cleaning your dataset for the movie prediction project. While exploring the dataset, you notice a strange pattern: there are some repeated, consecutive words occurring between the character at position 37 and the character at position 41. You decide to write a function to find out which movie reviews show this peculiarity, remembering that the ending position you specify is not inclusive. If you detect the word, you also want to change the string by replacing it with only one instance of the word.

# Complete the if-else statement following the instructions.

# The text of three movie reviews has been already saved in the variable movies. You can use print(movies) to view the variable in the IPython Shell.

Instructions
100 XP
Find if the substring actor occurs between the characters with index 37 and 41 inclusive. If it is not detected, print the statement Word not found.
Replace actor actor with the substring actor if actor occurs only two repeated times.
Replace actor actor actor with the substring actor if actor appears three repeated times.	

for movie in movies:
  	# Find if actor occurrs between 37 and 41
    if movie.find('actor', 37, 42) == -1:
        print("Word not found")
    # Count occurrences and replace two by one
    elif movie.count('actor') == 2:  
        print(movie.replace('actor actor', 'actor'))
    else:
        # Replace three occurrences by one
        print(movie.replace('actor actor actor', 'actor'))
		

## Where's the word?
# Before finishing cleaning your dataset, you want to check if a specific word occurs in the reviews. You noticed earlier a specific pattern in the strings. Now, you want to create a function to check if a word is present between characters with index 12, and 50, remembering that ending position is exclusive, and print out the lowest index where this word occurs. There are two methods to handle this situation. You want to see which one works best.

# The text of two movie reviews has been already saved in the variable movies. You can use print(movies) to view the variable in the IPython Shell.

# Find the index where money occurs between characters with index 12 and 50. If not found, the method should return -1.

for movie in movies:
  # Find the first occurrence of word
  print(movie.find('money', 12, 51))
  
 
 
 # Find the index where money occurs between characters with index 12 and 50. If not found, it should raise an error.
  
  for movie in movies:
  try:
    # Find the first occurrence of word
  	print(movie.index('money', 12, 51))
  except ValueError:
    print("substring not found")
	
	
## Replacing negations
# In order to keep working with your prediction project, your next task is to figure out how to handle negations that occur in your dataset. Some algorithms for prediction do not work well with negations, so a good way to handle this is to remove either not or n't, and to replace the next word by its antonym.

# Let's imagine that you have the string: The movie isn't good. You will need to remove n't and replace good for bad. This way, your string ends up being The movie is bad. You notice that in the first column of the dataset, you have a string that uses the word isn't followed by important.

# The text of this column has been already saved in the variable movies so you start working with it. You can use print(movies) to view it in the IPython Shell.
	
Instructions
100 XP
Replace the substring isn't for the word is.
Replace the substring important for the word insignificant.
Print out the result contained in the variable movies_antonym.

# Replace negations 
movies_no_negation = movies.replace("isn't", "is")

# Replace important
movies_antonym = movies_no_negation.replace("important", "insignificant")

# Print out
print(movies_antonym)


