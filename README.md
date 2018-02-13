# AutoSuggest

## How to run
NOTE: The model is hosted as a flask app. Please install flask before running the model
```bash
cd src
python ngram_preprocessing.py #Required only once
python run_ngram_model.py
```
This starts the autosuggest server. You can create any request as shown in the challenge requirements:
```
curl http://localhost:4000/autosuggest?q=When+did
```

## Preprocessing
 * Gather all sentences NOT originating from the customer (i.e agent created) [See discussion for why this choice and how it can be improved]
 * Append all sentences with start and end tokens
 * Use a Counter to save unique sentences and number of times they were used.
 * Build an inverted index of words -> sentences
 * Build an index of first words of all unique sentences (not used in the final pipeline)
 * Create a single continuous string of all unique sentences (separated by start and end tokens)
 * Store a list of sentences ordered by the number of hits (for lookup)
 
 
## Model 
 * I chose to use a simple bigram model
 * The model stores occurences of all pairs of words in the training data (Note that there is always a pair of words as we add a start symbol)
 * For every request, the sentence is split by space, start symbol is prefixed and the last two words of the sentence are considered. 
 * It tries to complete the last word of the sentence for every request. If the word is incomplete or there are other words possible with the same combination, probable words are generated. Note that If the word is already completed, it will be one of the probable words. 
 * For all the generated words, it generates possible strings. 
   For example: "I c" ---> "I can", "I can't" etc
 * Using the inverted index it generates a set of sentences that has all possible words in each candidate string
   Example: "I can" ---> 2,3,4
            "I can't" ---> 9,10, 11
            Sentence set = (2,3,4,9,10,11)
 * It then checks if all of these sentences starts with the string that was generated
 * All sentences are sorted in the order of their occurences in the training set. It thus returns the probable sentences ranked by number of hits
            
## Performance
The accuracy cannot not be verified without curating the dataset. However, it was verified that the user input string matches with the sentences generated (this is however expected)
Time taken to service a request:
 * Average total time : ~70ms
 * Data load time: ~45ms
 * Average prediction time: ~25ms
This is based on an average of 8 test inputs only. This was manually generated. A more accurate method would be to automate this. 

## Reasons for model choices
 * I started with bigram baseline. Turns out this works pretty well. There are some scaling issues that can be handled with efficient datastructures and/or approximation mechanisms.
 * While I did want to implement a more complicated language model with a generation based prediction (instead of retrieval based), I firmly believe that these models will be much slower. However, they can still be used for word completion/prediction after which the sentence can be completed using a retrieval based model
 
## Improvements if given more time

 * Implement an evaluation strategy to test how fast it is able to generate suggestions
 * **Implement context based ranking model**
 * Implement LSTM language model to suggest next word
 * Try a trigram model to evaluate possible improvements
 * Implement a model update mechanism based on usage statistics
 * Replace proper nouns (names) with a generic string

## Discussion

## Answers to challenge questions
1. How would you evaluate your autosuggest server? If you made another version, how would you compare the two to decide which is better?

   There are three important metrics in the autosuggest server:

   * **Accuracy of prediction**:  Accuracy is almost subjective here. We are enforcing the rule that all predicted sentences start with the the characters entered so far, so accuracy is pretty much guaranteed.
 
   * **Usefulness of ranking**:   Ranking can mostly be tested only at a user level. The prediction chosen by the user should be among the top k in the list of predictions returned by the model. '5' is a good value for k. The ranking can thus be refined over time as the user uses the system more often.
 
   * **Performance (time for prediction)**: The most important metric is the performance of the model. The predictions should update faster than the input. Typically an average user types at about 180 characters per minute (per https://www.livechatinc.com/typing-speed-test/#/). This translates to about 3 characters per second. This implies that a round trip for the request should take no longer than 300 milliseconds. Of course, there are users who do type faster than 3 characters per second and certain set of characters are obviously easier to type. So a safer ceiling will be about 150ms round trip time. 
  To evaluate this metric, a simple test script is needed that does the following:  
   - Generates a characters every 150ms 
   - Record the time taken for response
   - Average over a large set of requests
  

2. One way to improve the autosuggest server is to give topic-specific suggestions. How would you design an auto-categorization server? It should take a list of messages and return a TopicId. (Assume that every conversation in the training set has a TopicId).
This is a text classification problem. Given that topicID is present in the training data I would train a few different classifiers, evaluate their performance and choose the best one - 
 * A Naive Bayes with bag of words and TFIDF features 
 * A linear SVM with with bag of words and TFIDF features, 
 * A fully connected neural network with one hot vector and TFIDF features 
 * A LSTM to generate sentence embedding which then feeds into a fully connected neural network classfier  

3. How would you evaluate if your auto-categorization server is good?
Given that we have training data, the easiest method is to test for accuracy. 

4. Processing hundreds of millions of conversations for your autosuggest and auto-categorize models could take a very long time. How could you distribute the processing across multiple machines?
I tried to implement this with a multi process implementation (this would only be a simulation). But creating processes each time (given that flask does not allow you to maintain state between requests) was taking too long and chose to comment those statements out. This can of course be implemented better if there are stateful server
This is how I would distribute workload: 
 * The longest running part of the code is matching the input pattern with stored sentences. 
 * So, I would distribute the database of sentences across different machines
 * If the index is too big, we can also distribute the index across machines
 * All of the predicted patterns will be sent to the worker machines. The workers match the pattern with their cache of sentences.
 * The matched sentences are then sent back to the server to be served to the client. 
