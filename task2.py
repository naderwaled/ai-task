import PyPDF2
import google.generativeai as genai
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import time
# Make sure you have the NLTK data downloaded
nltk.download('punkt')

# Set your OpenAI API key
#genai.configure(api_key='AIzaSyChnJIZ9lTS73ZilftuIlS4H-Vbz4VsTzU')
genai.configure(api_key= 'AIzaSyBSKNWxj_bus8aS4IPfOONvLCCcdB1dECY')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def split_text(text, max_words=1500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) <= max_words:
            current_chunk += ' ' + sentence
            current_length += len(words)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = len(words)
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_chunk(model,chunk):
    prompt = f"Summarize the following text:\n\n{chunk}\n\nSummary:"
    response = model.generate_content(prompt)
    summary = response.text
    return summary

def summarize_book(model,chunks, summary_file):
    if os.path.exists(summary_file):
        print(f"Summaries already exist for this book. Loading from {summary_file}...")
        with open(summary_file, 'r', encoding='utf-8') as file:
            summaries = json.load(file)
    else:
        summaries = []
        for idx, chunk in enumerate(chunks):
            time.sleep(10)
            print(f"Summarizing chunk {idx+1}/{len(chunks)}...")
            summary = summarize_chunk(model,chunk)
            summaries.append(summary)
        # Save summaries to a file
        with open(summary_file, 'w', encoding='utf-8') as file:
            json.dump(summaries, file, ensure_ascii=False, indent=4)
        print(f"Summaries saved to {summary_file}.")
    return summaries


def find_relevant_chunks_ai(model,query, summaries, top_n=3):
    """
    Uses an AI model to find the most relevant summaries for a given query.

    Args:
        query (str): The search query.
        summaries (list): A list of summarized text chunks.
        top_n (int): Number of top relevant summaries to return.

    Returns:
        list: Indices of the top_n relevant summaries.
    """
    relevance_scores = []
    
    for idx, summary in enumerate(summaries):
        # Construct a prompt or API call for AI relevance scoring
        prompt = f"""
        Given the query:
        "{query}"

        Rank the relevance of the following summary on a scale of 0 to 10:
        "{summary}"

        Provide only the relevance score as a number:
        """
        
        # Use the AI model to get relevance score
        response = model.generate_content(prompt) # Adjust this based on the AI API
        relevance_score = float(response.text)  # Extract the score from the response
        
        relevance_scores.append((idx, relevance_score))
    
    # Sort summaries by relevance scores in descending order
    sorted_relevance = sorted(relevance_scores, key=lambda x: x[1], reverse=True)
    top_indices = [item[0] for item in sorted_relevance[:top_n]]
    
    return top_indices


def get_detailed_answer(model,chunks, relevant_indices, query):
    relevant_text = ' '.join([chunks[i] for i in relevant_indices])
    prompt = f"Answer the following question based on the provided text:\n\nText: {relevant_text}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    answer = response.text
    return answer

def main():
    book_name = "Harry Potter 1 - Harry Potter and the Sorcerer's Stone.pdf"
    summary_file = f"{book_name}_summaries.json"
    # Step 1: Read the book
    book_text = extract_text_from_pdf(book_name)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Step 2: Split the book into chunks
    chunks = split_text(book_text)

    # Step 3: Summarize each chunk
    summaries = summarize_book(model,chunks, summary_file)

    # Store summaries (optional: save to a file or database)
    # For this example, we'll keep them in memory

    # Step 4: Receive a query
    query = input("Enter your search query: ")

    # Step 5: Find relevant chunks
    relevant_indices = find_relevant_chunks_ai(model,query, summaries)

    # Step 6: Get detailed answer
    answer = get_detailed_answer(model,chunks, relevant_indices, query)

    print("\nDetailed Answer:")
    print(answer)

if __name__ == "__main__":
    main()
