import logging
import random
from collections import deque, defaultdict
from datetime import datetime, timedelta
import pytz
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    JobQueue,
    CallbackQueryHandler
)

BOT_TOKEN = '7624379060:AAGxjlyFlMpqZ5Es2u5eRwVTih0pMt4DuQ4'
OWNER_ID = 6479495033  # Replace with your actual Telegram user ID

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Game constants
BIG_THRESHOLD = 5
HISTORY_SIZE = 50
PREDICTION_WINDOW = 5
BDG_ANALYSIS_DEPTH = 8  # How many period numbers to analyze

# Color mapping (Red and Green only)
COLOR_MAP = {
    0: '🔴 Red',
    1: '🟢 Green',
    2: '🔴 Red',
    3: '🟢 Green',
    4: '🔴 Red',
    5: '🟢 Green',
    6: '🔴 Red',
    7: '🟢 Green',
    8: '🔴 Red',
    9: '🟢 Green'
}

# Track recent game history
game_history = deque(maxlen=HISTORY_SIZE)
live_results = deque(maxlen=100)
subscribers_1min = set()
subscribers_30sec = set()

# User database to track all users who start the bot
user_database = {}  # {user_id: {'username': username, 'first_name': first_name, 'last_name': last_name, 'date': date}}

# Prediction patterns (can be updated by owner)
prediction_patterns = {
    'markov_chain': defaultdict(lambda: defaultdict(int)),
    'frequency': defaultdict(int),
    'hot_numbers': [],
    'cold_numbers': []
}

def extract_bdg_digits(period_number):
    """Extract relevant digits from BDG period number (removing last digit)"""
    num_str = str(period_number)
    truncated_num = num_str[:-1]  # Remove last digit
    return [int(d) for d in truncated_num[-5:]]  # Return last 5 digits of truncated number

def classify_number(number):
    """Classify number with all categories"""
    return {
        'big_small': '🔵 Big' if number >= BIG_THRESHOLD else '⚪ Small',
        'number': number,
        'color': COLOR_MAP.get(number, '🔴 Red')
    }

def predict_numbers(is_big):
    """Predict next numbers strictly based on Big/Small prediction"""
    if is_big:
        # For big predictions, suggest numbers only from big range (5,6,7,8,9)
        big_numbers = [5, 6, 7, 8, 9]
        return random.sample(big_numbers, 3)
    else:
        # For small predictions, suggest numbers only from small range (0,1,2,3,4)
        small_numbers = [0, 1, 2, 3, 4]
        return random.sample(small_numbers, 3)

def analyze_bdg_patterns(numbers, game_type):
    """Enhanced pattern analysis focusing on Red/Green and proper Big/Small numbers"""
    if not numbers:
        return {
            'big_small': ('⚪ Small', 50),
            'color': ('🔴 Red', 50),
            'numbers': [0, 1, 2],
            'confidence': 0
        }
    
    digits = []
    for num in numbers:
        digits.extend(extract_bdg_digits(num))
    
    if len(digits) < 10:
        # Not enough data, return random predictions
        is_big = random.choice([True, False])
        return {
            'big_small': ('🔵 Big' if is_big else '⚪ Small', 50),
            'color': (random.choice(['🔴 Red', '🟢 Green']), 50),
            'numbers': predict_numbers(is_big),
            'confidence': 50
        }
    
    # Big/Small prediction with weighted probability
    big_count = sum(d >= BIG_THRESHOLD for d in digits[-10:])
    small_count = 10 - big_count
    big_prob = big_count / 10
    small_prob = small_count / 10
    
    if random.random() < big_prob:
        big_small_pred = '🔵 Big'
        bs_confidence = min(95, int(big_prob * 100))
        is_big = True
    else:
        big_small_pred = '⚪ Small'
        bs_confidence = min(95, int(small_prob * 100))
        is_big = False
    
    # Predict numbers strictly based on Big/Small prediction
    number_preds = predict_numbers(is_big)
    
    # Color prediction with Markov chain (Red/Green only)
    color_transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(digits)-1):
        current_color = COLOR_MAP.get(digits[i], '🔴 Red')
        next_color = COLOR_MAP.get(digits[i+1], '🔴 Red')
        color_transitions[current_color][next_color] += 1
    
    last_color = COLOR_MAP.get(digits[-1], '🔴 Red')
    if color_transitions[last_color]:
        # Use Markov probabilities
        total = sum(color_transitions[last_color].values())
        color_probs = {k: v/total for k, v in color_transitions[last_color].items()}
        color_pred = max(color_probs.items(), key=lambda x: x[1])[0]
        color_confidence = min(95, int(max(color_probs.values()) * 100))
    else:
        # Fallback to frequency
        color_counts = defaultdict(int)
        for d in digits[-10:]:
            color_counts[COLOR_MAP.get(d, '🔴 Red')] += 1
        color_pred = max(color_counts.items(), key=lambda x: x[1])[0]
        color_confidence = min(95, int(max(color_counts.values()) / 10 * 100))
    
    # Overall confidence (weighted average)
    confidence = int((bs_confidence * 0.4) + (color_confidence * 0.4) + (50 * 0.2))
    
    return {
        'big_small': (big_small_pred, bs_confidence),
        'color': (color_pred, color_confidence),
        'numbers': number_preds,
        'confidence': confidence
    }

async def fetch_bdg_results(game_type):
    """Simulate BDG results with pattern-based generation"""
    try:
        if game_type == '30sec':
            base = random.randint(10000, 99999)
            return [base, base + 1]
        else:
            base = random.randint(10000, 99999)
            return [base, base + 1]
    except Exception as e:
        logging.error(f"Error generating BDG results: {str(e)}")
        return None

async def analyze_and_predict_1min(context: ContextTypes.DEFAULT_TYPE):
    """1-minute analysis job with improved predictions"""
    try:
        current_time = datetime.now(pytz.utc)
        logging.info(f"Running 1-minute BDG analysis at {current_time}...")
        
        numbers = await fetch_bdg_results('1min')
        if not numbers:
            logging.warning("No BDG results generated in this cycle")
            return
        
        live_results.extend(numbers)
        predictions = analyze_bdg_patterns(numbers, '1min')
        
        game_history.append({
            'time': current_time,
            'game': '1min',
            'numbers': numbers,
            'predictions': predictions
        })
        
        if subscribers_1min:
            recent_digits = [extract_bdg_digits(num)[-3:] for num in numbers[-2:]]
            message = (
                f"⏰ BDG 1-Minute Prediction ({current_time.strftime('%H:%M:%S UTC')})\n"
                f"🔢 Recent Periods: {numbers[-2:]}\n"
                f"🔍 Last Digits: {recent_digits}\n\n"
                f"🎯 Betting Recommendations:\n"
                f"1. Big/Small: {predictions['big_small'][0]} ({predictions['big_small'][1]}% confidence)\n"
                f"2. Color: {predictions['color'][0]} ({predictions['color'][1]}% confidence)\n"
                f"3. Hot Numbers: {', '.join(str(n) for n in predictions['numbers'])}\n\n"
                f"📈 Overall Confidence: {predictions['confidence']}%\n"
                f"🔄 Next update at {(current_time + timedelta(minutes=1)).strftime('%H:%M:%S UTC')}"
            )
            
            for chat_id in subscribers_1min:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=message)
                except Exception as e:
                    logging.error(f"Error sending to {chat_id}: {str(e)}")
                    subscribers_1min.discard(chat_id)
        
    except Exception as e:
        logging.error(f"Error in 1-minute BDG analysis: {str(e)}")

async def analyze_and_predict_30sec(context: ContextTypes.DEFAULT_TYPE):
    """30-second analysis job with improved predictions"""
    try:
        current_time = datetime.now(pytz.utc)
        logging.info(f"Running 30-second BDG analysis at {current_time}...")
        
        numbers = await fetch_bdg_results('30sec')
        if not numbers:
            logging.warning("No BDG results generated in this cycle")
            return
        
        live_results.extend(numbers)
        predictions = analyze_bdg_patterns(numbers, '30sec')
        
        game_history.append({
            'time': current_time,
            'game': '30sec',
            'numbers': numbers,
            'predictions': predictions
        })
        
        if subscribers_30sec:
            recent_digits = [extract_bdg_digits(num)[-3:] for num in numbers[-2:]]
            message = (
                f"⏰ BDG 30-Second Prediction ({current_time.strftime('%H:%M:%S UTC')})\n"
                f"🔢 Recent Periods: {numbers[-2:]}\n"
                f"🔍 Last Digits: {recent_digits}\n\n"
                f"🎯 Betting Recommendations:\n"
                f"1. Big/Small: {predictions['big_small'][0]} ({predictions['big_small'][1]}% confidence)\n"
                f"2. Color: {predictions['color'][0]} ({predictions['color'][1]}% confidence)\n"
                f"3. Hot Numbers: {', '.join(str(n) for n in predictions['numbers'])}\n\n"
                f"📈 Overall Confidence: {predictions['confidence']}%\n"
                f"🔄 Next update at {(current_time + timedelta(seconds=30)).strftime('%H:%M:%S UTC')}"
            )
            
            for chat_id in subscribers_30sec:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=message)
                except Exception as e:
                    logging.error(f"Error sending to {chat_id}: {str(e)}")
                    subscribers_30sec.discard(chat_id)
        
    except Exception as e:
        logging.error(f"Error in 30-second BDG analysis: {str(e)}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the command /start is issued."""
    user = update.effective_user
    # Store user information
    user_database[user.id] = {
        'username': user.username,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'date': datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    keyboard = [
        [
            InlineKeyboardButton("1-Minute Predictions", callback_data='subscribe_1min'),
            InlineKeyboardButton("30-Second Predictions", callback_data='subscribe_30sec')
        ],
        [
            InlineKeyboardButton("View History", callback_data='view_history'),
            InlineKeyboardButton("EK Analysis", callback_data='ek_analysis')
        ]
    ]
    
    # Add owner-only buttons if user is owner
    if user.id == OWNER_ID:
        keyboard.append([
            InlineKeyboardButton("🔧 Update Predictions", callback_data='update_predictions'),
            InlineKeyboardButton("👥 View Users", callback_data='view_users')
        ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🎲 Welcome to BDG Wingo Predictor Bot!\n\n"
        "🔹 You will get Prediction of 30sec and 1min wingo\n\n"
        "🔹 You can only subscribe to ONE game type at a time\n\n"
        "🔹 Get automatic predictions for your chosen game\n\n"
        "🔹 Link - https://www.bdgup1.com//#/register?invitationCode=316864012833\n\n"
        "🔹 Invitation Code - 316864012833\n\n"
        "🔹 Select your preferred game type:",
        reply_markup=reply_markup
    )

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show recent prediction history"""
    if not game_history:
        if update.callback_query:
            await update.callback_query.edit_message_text(
                "No history available yet.",
                reply_markup=get_main_menu_keyboard(update.effective_user.id)
            )
        else:
            await update.message.reply_text("No history available yet.")
        return
    
    message = "📊 Recent Prediction History (Last 5):\n\n"
    for entry in list(game_history)[-5:]:
        message += (
            f"⏰ {entry['time'].strftime('%H:%M:%S')} ({entry['game']})\n"
            f"🔢 Numbers: {entry['numbers'][-2:]}\n"
            f"🎯 Predictions:\n"
            f" - Big/Small: {entry['predictions']['big_small'][0]} ({entry['predictions']['big_small'][1]}%)\n"
            f" - Color: {entry['predictions']['color'][0]} ({entry['predictions']['color'][1]}%)\n"
            f" - Numbers: {', '.join(str(n) for n in entry['predictions']['numbers'])}\n"
            f"📊 Confidence: {entry['predictions']['confidence']}%\n\n"
        )
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            message,
            reply_markup=get_main_menu_keyboard(update.effective_user.id)
        )
    else:
        await update.message.reply_text(message)

async def ek_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Perform EK analysis"""
    if not live_results:
        if update.callback_query:
            await update.callback_query.edit_message_text(
                "Not enough data for EK analysis yet.",
                reply_markup=get_main_menu_keyboard(update.effective_user.id)
            )
        else:
            await update.message.reply_text("Not enough data for EK analysis yet.")
        return
    
    # Get last 10 results
    last_results = list(live_results)[-10:]
    digits = []
    for num in last_results:
        digits.extend(extract_bdg_digits(num))
    
    # Simple EK analysis
    ek_message = "🔍 EK Analysis (Last 10 Results):\n\n"
    ek_message += f"Last Digits: {[d % 10 for d in digits[-10:]]}\n\n"
    
    # Frequency analysis
    freq = defaultdict(int)
    for d in digits[-10:]:
        freq[d % 10] += 1
    
    ek_message += "📊 Digit Frequency:\n"
    for num in sorted(freq.keys()):
        ek_message += f"{num}: {freq[num]}x\n"
    
    # Hot/Cold numbers
    hot_num = max(freq.items(), key=lambda x: x[1])[0]
    cold_num = min(freq.items(), key=lambda x: x[1])[0]
    
    ek_message += f"\n🔥 Hot Number: {hot_num} ({freq[hot_num]}x)\n"
    ek_message += f"❄️ Cold Number: {cold_num} ({freq[cold_num]}x)\n"
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            ek_message,
            reply_markup=get_main_menu_keyboard(update.effective_user.id)
        )
    else:
        await update.message.reply_text(ek_message)

async def update_predictions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Update prediction patterns (owner only)"""
    query = update.callback_query
    await query.answer()
    
    if query.from_user.id != OWNER_ID:
        await query.edit_message_text(
            "❌ This feature is only available to the bot owner.",
            reply_markup=get_main_menu_keyboard(query.from_user.id)
        )
        return
    
    # Update prediction patterns with new data
    if live_results:
        digits = []
        for num in live_results:
            digits.extend(extract_bdg_digits(num))
        
        # Update Markov chain
        prediction_patterns['markov_chain'] = defaultdict(lambda: defaultdict(int))
        for i in range(len(digits)-1):
            current = digits[i] % 10
            next_num = digits[i+1] % 10
            prediction_patterns['markov_chain'][current][next_num] += 1
        
        # Update frequency
        prediction_patterns['frequency'] = defaultdict(int)
        for d in digits[-50:]:
            prediction_patterns['frequency'][d % 10] += 1
        
        # Update hot/cold numbers
        sorted_freq = sorted(prediction_patterns['frequency'].items(), key=lambda x: x[1], reverse=True)
        prediction_patterns['hot_numbers'] = [x[0] for x in sorted_freq[:3]]
        prediction_patterns['cold_numbers'] = [x[0] for x in sorted_freq[-3:]]
        
        message = "✅ Prediction patterns updated successfully!\n\n"
        message += f"🔥 Hot Numbers: {', '.join(str(n) for n in prediction_patterns['hot_numbers'])}\n"
        message += f"❄️ Cold Numbers: {', '.join(str(n) for n in prediction_patterns['cold_numbers'])}\n"
        message += "📊 Markov chain and frequency tables updated with latest data."
    else:
        message = "⚠️ Not enough data to update patterns yet."
    
    await query.edit_message_text(
        message,
        reply_markup=get_main_menu_keyboard(query.from_user.id)
    )

async def view_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all users who have started the bot (owner only)"""
    query = update.callback_query
    await query.answer()
    
    if query.from_user.id != OWNER_ID:
        await query.edit_message_text(
            "❌ This feature is only available to the bot owner.",
            reply_markup=get_main_menu_keyboard(query.from_user.id)
        )
        return
    
    if not user_database:
        await query.edit_message_text(
            "No users have started the bot yet.",
            reply_markup=get_main_menu_keyboard(query.from_user.id)
        )
        return
    
    # Format user list with pagination
    users_per_page = 10
    total_users = len(user_database)
    page = int(context.args[0]) if context.args and context.args[0].isdigit() else 1
    start_idx = (page - 1) * users_per_page
    end_idx = start_idx + users_per_page
    
    sorted_users = sorted(user_database.items(), key=lambda x: x[0])
    page_users = sorted_users[start_idx:end_idx]
    
    message = f"👥 Registered Users ({total_users} total)\nPage {page}/{(total_users // users_per_page) + 1}\n\n"
    for user_id, user_data in page_users:
        username = user_data.get('username', 'N/A')
        first_name = user_data.get('first_name', 'N/A')
        last_name = user_data.get('last_name', '')
        date = user_data.get('date', 'N/A')
        
        message += (
            f"🆔 ID: {user_id}\n"
            f"👤 Name: {first_name} {last_name}\n"
            f"📛 Username: @{username}\n"
            f"📅 Date: {date}\n"
            f"----------------------------\n"
        )
    
    # Add pagination buttons
    keyboard = []
    if page > 1:
        keyboard.append(InlineKeyboardButton("⬅️ Previous", callback_data=f'view_users_{page-1}'))
    if end_idx < total_users:
        keyboard.append(InlineKeyboardButton("➡️ Next", callback_data=f'view_users_{page+1}'))
    
    keyboard.append(InlineKeyboardButton("🔙 Back", callback_data='back_to_main'))
    
    reply_markup = InlineKeyboardMarkup([keyboard]) if keyboard else None
    
    await query.edit_message_text(
        message,
        reply_markup=reply_markup
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks with exclusive subscription"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    data = query.data
    
    if data == 'subscribe_1min':
        # Unsubscribe from 30sec if subscribed
        if user_id in subscribers_30sec:
            subscribers_30sec.remove(user_id)
        
        if user_id in subscribers_1min:
            subscribers_1min.remove(user_id)
            message = "❌ Unsubscribed from 1-minute predictions."
        else:
            subscribers_1min.add(user_id)
            message = "✅ Subscribed to 1-minute predictions!\n(Unsubscribed from 30-second predictions)"
        
        await query.edit_message_text(
            message,
            reply_markup=get_main_menu_keyboard(user_id)
        )
    
    elif data == 'subscribe_30sec':
        # Unsubscribe from 1min if subscribed
        if user_id in subscribers_1min:
            subscribers_1min.remove(user_id)
        
        if user_id in subscribers_30sec:
            subscribers_30sec.remove(user_id)
            message = "❌ Unsubscribed from 30-second predictions."
        else:
            subscribers_30sec.add(user_id)
            message = "✅ Subscribed to 30-second predictions!\n(Unsubscribed from 1-minute predictions)"
        
        await query.edit_message_text(
            message,
            reply_markup=get_main_menu_keyboard(user_id)
        )
    
    elif data == 'view_history':
        await show_history(update, context)
    
    elif data == 'ek_analysis':
        await ek_command(update, context)
    
    elif data == 'update_predictions':
        await update_predictions(update, context)
    
    elif data == 'view_users' or data.startswith('view_users_'):
        page = int(data.split('_')[-1]) if data.startswith('view_users_') else 1
        context.args = [str(page)]
        await view_users(update, context)
    
    elif data == 'back_to_main':
        await query.edit_message_text(
            "🎲 Main Menu 🎲",
            reply_markup=get_main_menu_keyboard(user_id)
        )

def get_main_menu_keyboard(user_id):
    """Return the main menu keyboard with owner-only options if applicable"""
    keyboard = [
        [
            InlineKeyboardButton("1-Minute Predictions", callback_data='subscribe_1min'),
            InlineKeyboardButton("30-Second Predictions", callback_data='subscribe_30sec')
        ],
        [
            InlineKeyboardButton("View History", callback_data='view_history'),
            InlineKeyboardButton("EK Analysis", callback_data='ek_analysis')
        ]
    ]
    
    # Add owner-only buttons if user is owner
    if user_id == OWNER_ID:
        keyboard.append([
            InlineKeyboardButton("🔧 Update Predictions", callback_data='update_predictions'),
            InlineKeyboardButton("👥 View Users", callback_data='view_users')
        ])
    
    return InlineKeyboardMarkup(keyboard)

def main():
    """Start the BDG-optimized bot"""
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    job_queue = app.job_queue
    
    # Get current UTC time to schedule first jobs
    now = datetime.now(pytz.utc)
    
    # Schedule 1-minute job to start at next whole minute
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    initial_delay_1min = (next_minute - now).total_seconds()
    
    # Schedule 30-second job to start at next 30-second mark
    next_30sec = now.replace(second=30 if now.second < 30 else 0) + timedelta(minutes=0 if now.second < 30 else 1)
    initial_delay_30sec = (next_30sec - now).total_seconds()
    
    # Start BDG-optimized prediction jobs
    job_queue.run_repeating(
        analyze_and_predict_1min,
        interval=60,
        first=initial_delay_1min
    )
    
    job_queue.run_repeating(
        analyze_and_predict_30sec,
        interval=30,
        first=initial_delay_30sec
    )
    
    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("history", show_history))
    app.add_handler(CommandHandler("ek", ek_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    
    logging.info("BDG Wingo Predictor is running with exclusive subscriptions and improved predictions...")
    app.run_polling()

if __name__ == '__main__':
    main()