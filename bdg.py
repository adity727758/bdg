import logging
import random
from collections import deque
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    JobQueue,
    MessageHandler,
    filters,
)

BOT_TOKEN = '7624379060:AAGxjlyFlMpqZ5Es2u5eRwVTih0pMt4DuQ4'

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Game constants
BIG_THRESHOLD = 5
HISTORY_SIZE = 50
PREDICTION_WINDOW = 5
ANALYSIS_INTERVAL = 60  # seconds

# Track recent game history
game_history = deque(maxlen=HISTORY_SIZE)
live_results = deque(maxlen=100)
subscribers = set()

def classify_number(number):
    """Classify number as 'Big' or 'Small'"""
    return 'Big' if number >= BIG_THRESHOLD else 'Small'

def analyze_patterns(numbers):
    """Multiple pattern analysis methods with enhanced algorithms"""
    if not numbers:
        return "No data", 0
    
    classifications = [classify_number(num) for num in numbers]
    last_numbers = numbers[-PREDICTION_WINDOW:]
    
    # Method 1: Weighted moving average
    weights = [0.1, 0.15, 0.25, 0.3, 0.2][-len(last_numbers):]
    weighted_avg = sum(w*n for w, n in zip(weights, last_numbers)) / sum(weights)
    wma_pred = 'Big' if weighted_avg >= BIG_THRESHOLD else 'Small'
    
    # Method 2: Recent trend (last 5 with exponential weighting)
    recent = classifications[-5:]
    trend_weights = [0.1, 0.2, 0.3, 0.2, 0.2][-len(recent):]
    big_score = sum(w*(1 if c == 'Big' else 0) for w, c in zip(trend_weights, recent))
    small_score = sum(w*(1 if c == 'Small' else 0) for w, c in zip(trend_weights, recent))
    trend_pred = 'Big' if big_score > small_score else 'Small'
    
    # Method 3: Pattern recognition (repeating sequences)
    pattern_pred = None
    if len(numbers) >= 6:
        last_seq = numbers[-3:]
        for i in range(len(numbers)-6):
            if numbers[i:i+3] == last_seq and i+3 < len(numbers):
                pattern_pred = classify_number(numbers[i+3])
                break
    
    # Method 4: Statistical probability
    big_count = classifications.count('Big')
    small_count = classifications.count('Small')
    stat_pred = 'Big' if random.random() < (big_count / len(classifications)) else 'Small'
    
    # Combine predictions with confidence
    predictions = {
        'Weighted Average': wma_pred,
        'Trend Analysis': trend_pred,
        'Pattern Recognition': pattern_pred,
        'Statistical Probability': stat_pred
    }
    
    valid_preds = [p for p in predictions.values() if p is not None]
    if not valid_preds:
        return random.choice(['Big', 'Small']), 0
    
    # Calculate confidence (percentage of agreeing methods)
    final_pred = max(set(valid_preds), key=valid_preds.count)
    confidence = int((valid_preds.count(final_pred) / len(valid_preds)) * 100)
    
    return final_pred, confidence

async def fetch_live_results():
    """Simulate live results with random numbers"""
    try:
        # Generate random numbers that follow some patterns
        numbers = []
        for _ in range(10):
            # 70% chance to continue current pattern
            if numbers and random.random() < 0.7:
                last = numbers[-1]
                if last >= BIG_THRESHOLD:
                    numbers.append(random.randint(BIG_THRESHOLD, 9))
                else:
                    numbers.append(random.randint(0, BIG_THRESHOLD-1))
            else:
                numbers.append(random.randint(0, 9))
        
        return numbers
        
    except Exception as e:
        logging.error(f"Error generating results: {str(e)}")
        return None

async def analyze_and_predict(context: ContextTypes.DEFAULT_TYPE):
    """Periodic analysis job that runs every minute"""
    try:
        logging.info("Running automatic analysis...")
        
        numbers = await fetch_live_results()
        if not numbers:
            logging.warning("No results generated in this cycle")
            return
        
        live_results.extend(numbers)
        prediction, confidence = analyze_patterns(numbers)
        
        analysis_time = datetime.now(pytz.utc)
        game_history.append({
            'time': analysis_time,
            'numbers': numbers[-5:],  # Store only last 5 numbers
            'prediction': prediction,
            'confidence': confidence
        })
        
        if subscribers:
            message = (
                f"⏰ Automatic Prediction ({analysis_time.strftime('%H:%M:%S')})\n"
                f"🔢 Recent Numbers: {numbers[-5:]}\n"
                f"🔮 Next Outcome: {prediction}\n"
                f"📈 Confidence: {confidence}%\n"
                f"🔄 Next update in 1 minute"
            )
            
            for chat_id in subscribers:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=message)
                except Exception as e:
                    logging.error(f"Error sending to {chat_id}: {str(e)}")
                    subscribers.discard(chat_id)
        
    except Exception as e:
        logging.error(f"Error in automatic analysis: {str(e)}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message with instructions"""
    user = update.effective_user
    subscribers.add(user.id)
    
    await update.message.reply_text(
        f"🤖 Welcome {user.first_name} to Auto BDG Predictor!\n\n"
        "🔔 You'll now receive automatic predictions every minute\n\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/predict - Get current prediction\n"
        "/history - Show recent predictions\n"
        "/subscribe - Get live updates\n"
        "/unsubscribe - Stop live updates"
    )

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Subscribe to live updates"""
    user = update.effective_user
    subscribers.add(user.id)
    await update.message.reply_text(
        "✅ You'll receive automatic predictions every minute!"
    )

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Unsubscribe from live updates"""
    user = update.effective_user
    subscribers.discard(user.id)
    await update.message.reply_text(
        "❌ You won't receive automatic updates anymore.\n"
        "Use /subscribe to restart updates."
    )

async def get_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get current prediction"""
    if not live_results:
        await update.message.reply_text("Generating first prediction... Please wait a moment.")
        return
    
    prediction, confidence = analyze_patterns(list(live_results))
    last_numbers = list(live_results)[-5:]
    
    response = (
        f"📊 Current Prediction\n\n"
        f"🔢 Recent Numbers: {last_numbers}\n"
        f"🔮 Next Outcome: {prediction}\n"
        f"📈 Confidence: {confidence}%\n\n"
        f"🔄 Next auto-update in 1 minute"
    )
    
    await update.message.reply_text(response)

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show recent prediction history"""
    if not game_history:
        await update.message.reply_text("No history yet. First prediction coming soon!")
        return
    
    response = "📋 Prediction History (Last 10):\n\n"
    for i, entry in enumerate(reversed(game_history), 1):
        time_str = entry['time'].strftime('%H:%M:%S')
        nums = entry['numbers']
        pred = entry['prediction']
        conf = entry['confidence']
        response += f"{i}. {time_str} | Numbers: {nums} → {pred} ({conf}%)\n"
        if i >= 10:
            break
    
    await update.message.reply_text(response)

def main():
    """Start the bot with automatic predictions"""
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    job_queue = app.job_queue
    
    # Start automatic prediction job (every minute)
    job_queue.run_repeating(
        analyze_and_predict,
        interval=ANALYSIS_INTERVAL,
        first=5  # Start first prediction after 5 seconds
    )
    
    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", get_prediction))
    app.add_handler(CommandHandler("history", show_history))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    
    logging.info("Bot is running with automatic predictions every minute...")
    app.run_polling()

if __name__ == '__main__':
    main()
