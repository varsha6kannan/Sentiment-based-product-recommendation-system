from flask import Flask, request, render_template
from model import sentiment_recommender

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user ID as input
        user_id_input = request.form['username'].lower()

        try:
            # Call the sentiment-based recommendation function
            recommendations = sentiment_recommender(user_id_input)

            if recommendations is not None and not recommendations.empty:
                # Extract just the product names as a list
                product_names = recommendations['name'].tolist()
                return render_template("index.html", 
                                      output=product_names,
                                      user_input=user_id_input)
            else:
                return render_template("index.html", 
                                      message_display="No recommendations found for this user!")
        
        except Exception as e:
            print("Error:", e)
            return render_template("index.html", 
                                  message_display="User ID doesn't exist. Please provide a valid user!")

if __name__ == '__main__':
    app.run(debug=True)
