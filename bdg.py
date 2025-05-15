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
    CallbackQueryHandler,
    MessageHandler,
    filters
)

BOT_TOKEN = '7624379060:AAGxjlyFlMpqZ5Es2u5eRwVTih0pMt4DuQ4'
OWNER_ID = 6479495033  # Replace with your actual Telegram user ID

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Enhanced game constants
BIG_THRESHOLD = 5
HISTORY_SIZE = 100  # Increased history size
PREDICTION_WINDOW = 10  # Increased window for better pattern recognition
ANALYSIS_DEPTH = 20  # Deeper analysis for better predictions

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

# Platform links
PLATFORM_LINKS = {
    'bdg': {
        'name': 'BDG Games',
        'url': 'https://bdggame.com',
        'description': 'Official BDG Games website'
    },
    'tc': {
        'name': 'TC Lottery',
        'url': 'https://tclottery.in',
        'description': 'Official TC Lottery website'
    },
    'mumbai': {
        'name': 'Big Mumbai',
        'url': 'https://bigmumbai.com',
        'description': 'Official Big Mumbai website'
    },
    'daman': {
        'name': 'Daman Games',
        'url': 'https://damangames.com',
        'description': 'Official Daman Games website'
    }
}

# Track recent game history
game_history = deque(maxlen=HISTORY_SIZE)
live_results = deque(maxlen=200)  # Increased live results storage
subscribers = {
    'bdg': {
        '1min': set(),
        '30sec': set()
    },
    'tc': {
        '1min': set(),
        '30sec': set()
    },
    'mumbai': {
        '1min': set(),
        '30sec': set()
    },
    'daman': {
        '1min': set(),
        '30sec': set()
    }
}

# Enhanced user database
user_database = {}

# Advanced prediction patterns
prediction_patterns = {
    'markov_chain': defaultdict(lambda: defaultdict(int)),
    'frequency': defaultdict(int),
    'color_transitions': defaultdict(lambda: defaultdict(int)),
    'hot_numbers': [],
    'cold_numbers': [],
    'streaks': {
        'big': {'current': 0, 'max': 0},
        'small': {'current': 0, 'max': 0},
        'red': {'current': 0, 'max': 0},
        'green': {'current': 0, 'max': 0}
    },
    'last_updated': datetime.now(pytz.utc)
}

# Track URL editing state
url_editing_state = {}

def extract_bdg_digits(period_number):
    num_str = str(period_number)
    truncated_num = num_str[:-1]
    return [int(d) for d in truncated_num[-5:]]

def classify_number(number):
    return {
        'big_small': '🔵 Big' if number >= BIG_THRESHOLD else '⚪ Small',
        'number': number,
        'color': COLOR_MAP.get(number, '🔴 Red')
    }

def predict_numbers(is_big):
    # Base numbers based on big/small
    base_numbers = list(range(5, 10)) if is_big else list(range(0, 5))
    
    # If we have hot numbers from analysis, use them
    if prediction_patterns['hot_numbers']:
        hot_in_range = [n for n in prediction_patterns['hot_numbers'] if (n >= 5 if is_big else n < 5)]
        if hot_in_range:
            base_numbers = hot_in_range + [n for n in base_numbers if n not in hot_in_range]
    
    # Ensure we have enough numbers
    if len(base_numbers) < 3:
        base_numbers.extend(random.sample([n for n in range(10) if n not in base_numbers], 3 - len(base_numbers)))
    
    return random.sample(base_numbers, min(3, len(base_numbers)))

def analyze_patterns(numbers, game_type):
    if not numbers:
        return {
            'big_small': ('⚪ Small', 50),
            'color': ('🔴 Red', 50),
            'numbers': [0, 1, 2],
            'confidence': 0,
            'hot_numbers': [],
            'cold_numbers': []
        }
    
    digits = []
    for num in numbers:
        digits.extend(extract_bdg_digits(num))
    
    if len(digits) < 10:
        is_big = random.choice([True, False])
        return {
            'big_small': ('🔵 Big' if is_big else '⚪ Small', 50),
            'color': (random.choice(['🔴 Red', '🟢 Green']), 50),
            'numbers': predict_numbers(is_big),
            'confidence': 50,
            'hot_numbers': [],
            'cold_numbers': []
        }
    
    # Update streaks
    update_streaks(digits)
    
    # Enhanced Big/Small analysis with weighted history
    window_size = min(ANALYSIS_DEPTH, len(digits))
    weighted_big = 0
    weighted_small = 0
    
    for i in range(window_size):
        weight = (window_size - i) / window_size  # More weight to recent numbers
        if digits[-i-1] >= BIG_THRESHOLD:
            weighted_big += weight
        else:
            weighted_small += weight
    
    big_prob = weighted_big / (weighted_big + weighted_small)
    small_prob = 1 - big_prob
    
    # Dynamic threshold adjustment
    dynamic_threshold = BIG_THRESHOLD
    if big_prob > 0.7:
        dynamic_threshold = max(4, BIG_THRESHOLD - 1)
    elif small_prob > 0.7:
        dynamic_threshold = min(6, BIG_THRESHOLD + 1)
    
    # Final big/small prediction with dynamic threshold
    recent_avg = sum(digits[-5:]) / 5 if len(digits) >= 5 else BIG_THRESHOLD
    is_big = recent_avg >= dynamic_threshold
    big_small_pred = '🔵 Big' if is_big else '⚪ Small'
    bs_confidence = min(95, int(max(big_prob, small_prob) * 100))
    
    # Enhanced color prediction
    color_pred, color_confidence = predict_color(digits)
    
    # Number prediction with hot/cold analysis
    freq = defaultdict(int)
    for i, d in enumerate(digits[-ANALYSIS_DEPTH:]):
        freq[d % 10] += (ANALYSIS_DEPTH - i) * 0.2  # Weight recent numbers more
    
    hot_numbers = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
    cold_numbers = sorted(freq.items(), key=lambda x: x[1])[:3]
    
    predicted_numbers = list({n for n, _ in hot_numbers})
    if is_big:
        predicted_numbers.extend([n for n in range(5, 10) if n not in predicted_numbers])
    else:
        predicted_numbers.extend([n for n in range(0, 5) if n not in predicted_numbers])
    
    predicted_numbers = list(dict.fromkeys(predicted_numbers))[:3]
    
    # Update global patterns
    update_global_patterns(digits, hot_numbers, cold_numbers)
    
    # Calculate overall confidence
    confidence_factors = [
        bs_confidence * 0.4,
        color_confidence * 0.4,
        (sum(f for _, f in hot_numbers[:3]) / (3 * max(1, max(freq.values())))) * 20
    ]
    
    confidence = min(95, int(sum(confidence_factors)))
    
    return {
        'big_small': (big_small_pred, bs_confidence),
        'color': (color_pred, color_confidence),
        'numbers': predicted_numbers,
        'confidence': confidence,
        'hot_numbers': [n for n, _ in hot_numbers],
        'cold_numbers': [n for n, _ in cold_numbers]
    }

def update_streaks(digits):
    # Reset streaks if needed
    if (datetime.now(pytz.utc) - prediction_patterns['last_updated']).total_seconds() > 3600:
        for streak in prediction_patterns['streaks'].values():
            streak['current'] = 0
    
    # Update big/small streaks
    current_bs = 'big' if digits[-1] >= BIG_THRESHOLD else 'small'
    opposite_bs = 'small' if current_bs == 'big' else 'big'
    
    if len(digits) > 1:
        prev_bs = 'big' if digits[-2] >= BIG_THRESHOLD else 'small'
        if current_bs == prev_bs:
            prediction_patterns['streaks'][current_bs]['current'] += 1
            prediction_patterns['streaks'][current_bs]['max'] = max(
                prediction_patterns['streaks'][current_bs]['max'],
                prediction_patterns['streaks'][current_bs]['current']
            )
        else:
            prediction_patterns['streaks'][current_bs]['current'] = 1
            prediction_patterns['streaks'][opposite_bs]['current'] = 0
    
    # Update color streaks
    current_color = 'red' if COLOR_MAP.get(digits[-1], '').startswith('🔴') else 'green'
    opposite_color = 'green' if current_color == 'red' else 'red'
    
    if len(digits) > 1:
        prev_color = 'red' if COLOR_MAP.get(digits[-2], '').startswith('🔴') else 'green'
        if current_color == prev_color:
            prediction_patterns['streaks'][current_color]['current'] += 1
            prediction_patterns['streaks'][current_color]['max'] = max(
                prediction_patterns['streaks'][current_color]['max'],
                prediction_patterns['streaks'][current_color]['current']
            )
        else:
            prediction_patterns['streaks'][current_color]['current'] = 1
            prediction_patterns['streaks'][opposite_color]['current'] = 0
    
    prediction_patterns['last_updated'] = datetime.now(pytz.utc)

def predict_color(digits):
    last_color = COLOR_MAP.get(digits[-1], '🔴 Red')
    
    # Calculate color probabilities with smoothing and streak adjustment
    if prediction_patterns['color_transitions'][last_color]:
        total = sum(prediction_patterns['color_transitions'][last_color].values())
        color_probs = {
            '🔴 Red': (prediction_patterns['color_transitions'][last_color]['🔴 Red'] + 1) / (total + 2),
            '🟢 Green': (prediction_patterns['color_transitions'][last_color]['🟢 Green'] + 1) / (total + 2)
        }
    else:
        total_colors = sum(sum(v.values()) for v in prediction_patterns['color_transitions'].values())
        color_probs = {
            '🔴 Red': 0.5,
            '🟢 Green': 0.5
        }
    
    # Adjust for streaks
    current_streak = prediction_patterns['streaks']['red']['current'] if last_color == '🔴 Red' else prediction_patterns['streaks']['green']['current']
    if current_streak >= 3:
        # Reduce probability of continuing long streaks
        streak_color = '🔴 Red' if last_color == '🔴 Red' else '🟢 Green'
        color_probs[streak_color] *= max(0.3, 1 - (current_streak * 0.15))
        color_probs = {k: v/sum(color_probs.values()) for k, v in color_probs.items()}
    
    color_pred = max(color_probs.items(), key=lambda x: x[1])[0]
    color_confidence = min(95, int(max(color_probs.values()) * 100))
    
    return color_pred, color_confidence

def update_global_patterns(digits, hot_numbers, cold_numbers):
    # Update Markov chain for numbers
    for i in range(len(digits)-1):
        current = digits[i] % 10
        next_num = digits[i+1] % 10
        prediction_patterns['markov_chain'][current][next_num] += 1
    
    # Update frequency counts
    for d in digits[-ANALYSIS_DEPTH:]:
        prediction_patterns['frequency'][d % 10] += 1
    
    # Update color transitions
    for i in range(len(digits)-1):
        current_color = COLOR_MAP.get(digits[i], '🔴 Red')
        next_color = COLOR_MAP.get(digits[i+1], '🔴 Red')
        prediction_patterns['color_transitions'][current_color][next_color] += 1
    
    # Update hot/cold numbers
    prediction_patterns['hot_numbers'] = [n for n, _ in hot_numbers]
    prediction_patterns['cold_numbers'] = [n for n, _ in cold_numbers]

async def fetch_bdg_results(game_type):
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
    try:
        current_time = datetime.now(pytz.utc)
        logging.info(f"Running 1-minute BDG analysis at {current_time}...")
        
        numbers = await fetch_bdg_results('1min')
        if not numbers:
            logging.warning("No BDG results generated in this cycle")
            return
        
        live_results.extend(numbers)
        predictions = analyze_patterns(numbers, '1min')
        
        game_history.append({
            'time': current_time,
            'game': '1min',
            'numbers': numbers,
            'predictions': predictions,
            'platform': 'bdg'
        })
        
        for platform in ['bdg', 'tc', 'mumbai', 'daman']:
            if subscribers[platform]['1min']:
                recent_digits = [extract_bdg_digits(num)[-3:] for num in numbers[-2:]]
                message = (
                    f"⏰ {platform.upper()} 1-Minute Prediction ({current_time.strftime('%H:%M:%S UTC')})\n"
                    f"🔢 Recent Periods: {numbers[-2:]}\n"
                    f"🔍 Last Digits: {recent_digits}\n\n"
                    f"🎯 Betting Recommendations:\n"
                    f"1. Big/Small: {predictions['big_small'][0]} ({predictions['big_small'][1]}% confidence)\n"
                    f"2. Color: {predictions['color'][0]} ({predictions['color'][1]}% confidence)\n"
                    f"3. Hot Numbers: {', '.join(str(n) for n in predictions['numbers'])}\n"
                    f"4. Cold Numbers: {', '.join(str(n) for n in predictions['cold_numbers'])}\n\n"
                    f"📈 Overall Confidence: {predictions['confidence']}%\n"
                    f"🔄 Next update at {(current_time + timedelta(minutes=1)).strftime('%H:%M:%S UTC')}"
                )
                
                for chat_id in subscribers[platform]['1min']:
                    try:
                        await context.bot.send_message(chat_id=chat_id, text=message)
                    except Exception as e:
                        logging.error(f"Error sending to {chat_id}: {str(e)}")
                        subscribers[platform]['1min'].discard(chat_id)
        
    except Exception as e:
        logging.error(f"Error in 1-minute BDG analysis: {str(e)}")

async def analyze_and_predict_30sec(context: ContextTypes.DEFAULT_TYPE):
    try:
        current_time = datetime.now(pytz.utc)
        logging.info(f"Running 30-second BDG analysis at {current_time}...")
        
        numbers = await fetch_bdg_results('30sec')
        if not numbers:
            logging.warning("No BDG results generated in this cycle")
            return
        
        live_results.extend(numbers)
        predictions = analyze_patterns(numbers, '30sec')
        
        game_history.append({
            'time': current_time,
            'game': '30sec',
            'numbers': numbers,
            'predictions': predictions,
            'platform': 'bdg'
        })
        
        for platform in ['bdg', 'tc', 'mumbai', 'daman']:
            if subscribers[platform]['30sec']:
                recent_digits = [extract_bdg_digits(num)[-3:] for num in numbers[-2:]]
                message = (
                    f"⏰ {platform.upper()} 30-Second Prediction ({current_time.strftime('%H:%M:%S UTC')})\n"
                    f"🔢 Recent Periods: {numbers[-2:]}\n"
                    f"🔍 Last Digits: {recent_digits}\n\n"
                    f"🎯 Betting Recommendations:\n"
                    f"1. Big/Small: {predictions['big_small'][0]} ({predictions['big_small'][1]}% confidence)\n"
                    f"2. Color: {predictions['color'][0]} ({predictions['color'][1]}% confidence)\n"
                    f"3. Hot Numbers: {', '.join(str(n) for n in predictions['numbers'])}\n"
                    f"4. Cold Numbers: {', '.join(str(n) for n in predictions['cold_numbers'])}\n\n"
                    f"📈 Overall Confidence: {predictions['confidence']}%\n"
                    f"🔄 Next update at {(current_time + timedelta(seconds=30)).strftime('%H:%M:%S UTC')}"
                )
                
                for chat_id in subscribers[platform]['30sec']:
                    try:
                        await context.bot.send_message(chat_id=chat_id, text=message)
                    except Exception as e:
                        logging.error(f"Error sending to {chat_id}: {str(e)}")
                        subscribers[platform]['30sec'].discard(chat_id)
        
    except Exception as e:
        logging.error(f"Error in 30-second BDG analysis: {str(e)}")

async def perform_prediction_update(update: Update, platform: str):
    query = update.callback_query
    await query.answer()
    
    if query.from_user.id != OWNER_ID:
        await query.edit_message_text(
            "❌ This feature is only available to the bot owner.",
            reply_markup=get_main_menu_keyboard(query.from_user.id)
        )
        return
    
    current_time = datetime.now(pytz.utc)
    logging.info(f"Running manual prediction update for {platform} at {current_time}")
    
    game_types = ['1min', '30sec']
    messages = []
    
    for game_type in game_types:
        numbers = await fetch_bdg_results(game_type)
        if not numbers:
            messages.append(f"⚠️ Could not generate results for {platform.upper()} {game_type}")
            continue
        
        predictions = analyze_patterns(numbers, game_type)
        
        game_history.append({
            'time': current_time,
            'game': game_type,
            'numbers': numbers,
            'predictions': predictions,
            'platform': platform
        })
        
        recent_digits = [extract_bdg_digits(num)[-3:] for num in numbers[-2:]]
        message = (
            f"🔄 Manual Update: {platform.upper()} {game_type} Prediction ({current_time.strftime('%H:%M:%S UTC')})\n"
            f"🔢 Recent Periods: {numbers[-2:]}\n"
            f"🔍 Last Digits: {recent_digits}\n\n"
            f"🎯 Updated Betting Recommendations:\n"
            f"1. Big/Small: {predictions['big_small'][0]} ({predictions['big_small'][1]}% confidence)\n"
            f"2. Color: {predictions['color'][0]} ({predictions['color'][1]}% confidence)\n"
            f"3. Hot Numbers: {', '.join(str(n) for n in predictions['hot_numbers'])}\n"
            f"4. Cold Numbers: {', '.join(str(n) for n in predictions['cold_numbers'])}\n\n"
            f"📈 Overall Confidence: {predictions['confidence']}%"
        )
        messages.append(message)
        
        for chat_id in subscribers[platform][game_type]:
            try:
                await query.bot.send_message(chat_id=chat_id, text=message)
            except Exception as e:
                logging.error(f"Error sending to {chat_id}: {str(e)}")
                subscribers[platform][game_type].discard(chat_id)
    
    response_message = "\n\n".join(messages)
    await query.edit_message_text(
        response_message,
        reply_markup=get_main_menu_keyboard(query.from_user.id)
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_database[user.id] = {
        'username': user.username,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'date': datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    await update.message.reply_text(
        "🎲 Welcome to Multi-Platform Lottery Predictor Bot!\n\n"
        "🔹 Get predictions for multiple lottery platforms\n"
        "🔹 Choose your preferred platform and game type\n"
        "🔹 Receive automatic predictions for your selection\n\n"
        "Please select a platform:",
        reply_markup=get_main_menu_keyboard(user.id)
    )

async def show_platform_links(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    message = "🔗 Platform Links:\n\n"
    
    for platform, info in PLATFORM_LINKS.items():
        message += (
            f"🏆 {info['name']}\n"
            f"🌐 {info['url']}\n"
            f"📝 {info['description']}\n\n"
        )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='back_to_main')]]
    
    if user_id == OWNER_ID:
        keyboard.insert(0, [InlineKeyboardButton("✏️ Edit Links", callback_data='edit_links')])
    
    await query.edit_message_text(
        message,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def edit_links_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    keyboard = [
        [InlineKeyboardButton("🎲 Edit BDG Link", callback_data='edit_bdg_link')],
        [InlineKeyboardButton("🎰 Edit TC Link", callback_data='edit_tc_link')],
        [InlineKeyboardButton("🏙️ Edit Mumbai Link", callback_data='edit_mumbai_link')],
        [InlineKeyboardButton("🌉 Edit Daman Link", callback_data='edit_daman_link')],
        [InlineKeyboardButton("🔙 Back", callback_data='platform_links')]
    ]
    
    await query.edit_message_text(
        "✏️ Select which platform link you want to edit:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def prompt_for_new_url(update: Update, platform: str):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    url_editing_state[user_id] = {'platform': platform, 'step': 'waiting_for_url'}
    
    await query.edit_message_text(
        f"Please enter the new URL for {PLATFORM_LINKS[platform]['name']}:\n\n"
        f"Current URL: {PLATFORM_LINKS[platform]['url']}\n\n"
        "Type 'cancel' to abort.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data='cancel_edit')]])
    )

async def handle_url_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    
    if user_id not in url_editing_state:
        return
    
    if url_editing_state[user_id]['step'] != 'waiting_for_url':
        return
    
    new_url = update.message.text.strip()
    
    if new_url.lower() == 'cancel':
        del url_editing_state[user_id]
        await update.message.reply_text(
            "URL update cancelled.",
            reply_markup=get_main_menu_keyboard(user_id)
        )
        return
    
    platform = url_editing_state[user_id]['platform']
    PLATFORM_LINKS[platform]['url'] = new_url
    del url_editing_state[user_id]
    
    await update.message.reply_text(
        f"✅ {PLATFORM_LINKS[platform]['name']} URL updated to:\n{new_url}",
        reply_markup=get_main_menu_keyboard(user_id)
    )

async def cancel_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    if user_id in url_editing_state:
        del url_editing_state[user_id]
    
    await query.edit_message_text(
        "URL update cancelled.",
        reply_markup=get_main_menu_keyboard(user_id)
    )

async def show_platform_menu(update: Update, platform: str):
    query = update.callback_query
    await query.answer()
    
    platform_name = {
        'bdg': '🎲 BDG Games',
        'tc': '🎰 TC Lottery',
        'mumbai': '🏙️ Big Mumbai',
        'daman': '🌉 Daman Games'
    }.get(platform, platform)
    
    keyboard = [
        [
            InlineKeyboardButton("1-Minute Predictions", callback_data=f'subscribe_{platform}_1min'),
            InlineKeyboardButton("30-Second Predictions", callback_data=f'subscribe_{platform}_30sec')
        ],
        [InlineKeyboardButton("🔙 Back to Main Menu", callback_data='back_to_main')]
    ]
    
    await query.edit_message_text(
        f"{platform_name} - Select Game Type:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            f"⏰ {entry['time'].strftime('%H:%M:%S')} ({entry['game']}) - {entry.get('platform', 'BDG').upper()}\n"
            f"🔢 Numbers: {entry['numbers'][-2:]}\n"
            f"🎯 Predictions:\n"
            f" - Big/Small: {entry['predictions']['big_small'][0]} ({entry['predictions']['big_small'][1]}%)\n"
            f" - Color: {entry['predictions']['color'][0]} ({entry['predictions']['color'][1]}%)\n"
            f" - Numbers: {', '.join(str(n) for n in entry['predictions']['numbers'])}\n"
            f" - Hot Numbers: {', '.join(str(n) for n in entry['predictions'].get('hot_numbers', []))}\n"
            f" - Cold Numbers: {', '.join(str(n) for n in entry['predictions'].get('cold_numbers', []))}\n"
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
    if not live_results:
        if update.callback_query:
            await update.callback_query.edit_message_text(
                "Not enough data for EK analysis yet.",
                reply_markup=get_main_menu_keyboard(update.effective_user.id)
            )
        else:
            await update.message.reply_text("Not enough data for EK analysis yet.")
        return
    
    last_results = list(live_results)[-10:]
    digits = []
    for num in last_results:
        digits.extend(extract_bdg_digits(num))
    
    ek_message = "🔍 EK Analysis (Last 10 Results):\n\n"
    ek_message += f"Last Digits: {[d % 10 for d in digits[-10:]]}\n\n"
    
    freq = defaultdict(int)
    for d in digits[-10:]:
        freq[d % 10] += 1
    
    ek_message += "📊 Digit Frequency:\n"
    for num in sorted(freq.keys()):
        ek_message += f"{num}: {freq[num]}x\n"
    
    hot_num = max(freq.items(), key=lambda x: x[1])[0]
    cold_num = min(freq.items(), key=lambda x: x[1])[0]
    
    ek_message += f"\n🔥 Hot Number: {hot_num} ({freq[hot_num]}x)\n"
    ek_message += f"❄️ Cold Number: {cold_num} ({freq[cold_num]}x)\n"
    
    # Add streak information
    bs_streak = prediction_patterns['streaks']['big'] if digits[-1] >= BIG_THRESHOLD else prediction_patterns['streaks']['small']
    color_streak = prediction_patterns['streaks']['red'] if COLOR_MAP.get(digits[-1], '').startswith('🔴') else prediction_patterns['streaks']['green']
    
    ek_message += (
        f"\n📈 Current Streaks:\n"
        f"Big/Small: {bs_streak['current']} (Max: {bs_streak['max']})\n"
        f"Color: {color_streak['current']} (Max: {color_streak['max']})"
    )
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            ek_message,
            reply_markup=get_main_menu_keyboard(update.effective_user.id)
        )
    else:
        await update.message.reply_text(ek_message)

async def update_predictions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.from_user.id != OWNER_ID:
        await query.edit_message_text(
            "❌ This feature is only available to the bot owner.",
            reply_markup=get_main_menu_keyboard(query.from_user.id)
        )
        return
    
    latest_results = list(live_results)[-ANALYSIS_DEPTH:] if live_results else []
    
    if not latest_results:
        await query.edit_message_text(
            "⚠️ Not enough data to update patterns yet.",
            reply_markup=get_main_menu_keyboard(query.from_user.id)
        )
        return
    
    digits = []
    for num in latest_results:
        digits.extend(extract_bdg_digits(num))
    
    # Perform comprehensive analysis
    predictions = analyze_patterns(latest_results, 'manual_update')
    
    # Prepare detailed analysis message
    message = "🔄 Updated Prediction Patterns:\n\n"
    message += "📊 Big/Small Analysis:\n"
    message += f"🔵 Big: {predictions['big_small'][1]}% (Current streak: {prediction_patterns['streaks']['big']['current']})\n"
    message += f"⚪ Small: {100 - predictions['big_small'][1]}% (Current streak: {prediction_patterns['streaks']['small']['current']})\n\n"
    
    message += "🎨 Color Analysis:\n"
    message += f"🔴 Red: {predictions['color'][1]}% (Current streak: {prediction_patterns['streaks']['red']['current']})\n"
    message += f"🟢 Green: {100 - predictions['color'][1]}% (Current streak: {prediction_patterns['streaks']['green']['current']})\n\n"
    
    message += "🔢 Number Analysis:\n"
    message += f"🔥 Hot Numbers: {', '.join(str(n) for n in predictions['hot_numbers'])}\n"
    message += f"❄️ Cold Numbers: {', '.join(str(n) for n in predictions['cold_numbers'])}\n\n"
    
    message += "📈 Current Recommendations:\n"
    if predictions['big_small'][1] > 60:
        message += "➡️ Strong Big trend detected\n"
    elif predictions['big_small'][1] < 40:
        message += "➡️ Strong Small trend detected\n"
    else:
        message += "➡️ Neutral trend - consider both options\n"
    
    if predictions['color'][1] > 60:
        message += f"➡️ Strong {predictions['color'][0]} trend detected\n"
    elif predictions['color'][1] < 40:
        opp_color = '🟢 Green' if predictions['color'][0] == '🔴 Red' else '🔴 Red'
        message += f"➡️ Strong {opp_color} trend detected\n"
    else:
        message += "➡️ Neutral color trend\n"
    
    if prediction_patterns['streaks']['big']['current'] >= 3:
        message += f"⚠️ Big streak of {prediction_patterns['streaks']['big']['current']} - consider Small soon\n"
    if prediction_patterns['streaks']['small']['current'] >= 3:
        message += f"⚠️ Small streak of {prediction_patterns['streaks']['small']['current']} - consider Big soon\n"
    if prediction_patterns['streaks']['red']['current'] >= 3:
        message += f"⚠️ Red streak of {prediction_patterns['streaks']['red']['current']} - consider Green soon\n"
    if prediction_patterns['streaks']['green']['current'] >= 3:
        message += f"⚠️ Green streak of {prediction_patterns['streaks']['green']['current']} - consider Red soon\n"
    
    keyboard = [
        [InlineKeyboardButton("🔙 Back to Main Menu", callback_data='back_to_main')],
        [InlineKeyboardButton("🔄 Update Again", callback_data='update_predictions')]
    ]
    
    await query.edit_message_text(
        message,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def view_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    data = query.data
    
    if data.startswith('platform_'):
        if data == 'platform_links':
            await show_platform_links(update, context)
        else:
            platform = data.split('_')[1]
            await show_platform_menu(update, platform)
    
    elif data == 'edit_links':
        await edit_links_menu(update, context)
    
    elif data.startswith('edit_') and data.endswith('_link'):
        platform = data.split('_')[1]
        await prompt_for_new_url(update, platform)
    
    elif data == 'cancel_edit':
        await cancel_edit(update, context)
    
    elif data.startswith('subscribe_'):
        parts = data.split('_')
        if len(parts) == 3:
            platform = parts[1]
            game_type = parts[2]
            
            for plat in subscribers:
                for gt in subscribers[plat]:
                    if user_id in subscribers[plat][gt]:
                        subscribers[plat][gt].remove(user_id)
            
            subscribers[platform][game_type].add(user_id)
            
            platform_name = {
                'bdg': 'BDG Games',
                'tc': 'TC Lottery',
                'mumbai': 'Big Mumbai',
                'daman': 'Daman Games'
            }.get(platform, platform)
            
            message = (
                f"✅ Successfully subscribed to {platform_name} {game_type} predictions!\n\n"
                f"⚠️ You have been automatically unsubscribed from all other games.\n\n"
                f"You will now receive {game_type} predictions for {platform_name}."
            )
            
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
    
    elif data.startswith('update_'):
        platform = data.split('_')[1]
        await perform_prediction_update(update, platform)
    
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
    keyboard = [
        [InlineKeyboardButton("🎲 BDG Games", callback_data='platform_bdg')],
        [InlineKeyboardButton("🎰 TC Lottery", callback_data='platform_tc')],
        [InlineKeyboardButton("🏙️ Big Mumbai", callback_data='platform_mumbai')],
        [InlineKeyboardButton("🌉 Daman Games", callback_data='platform_daman')],
        [InlineKeyboardButton("🔗 Platform Links", callback_data='platform_links')],
        [InlineKeyboardButton("📊 View History", callback_data='view_history')],
        [InlineKeyboardButton("🔍 EK Analysis", callback_data='ek_analysis')]
    ]
    
    if user_id == OWNER_ID:
        owner_buttons = [
            [
                InlineKeyboardButton("👥 View Users", callback_data='view_users')
            ],
            [InlineKeyboardButton("🔧 Global Update", callback_data='update_predictions')]
        ]
        keyboard.extend(owner_buttons)
    
    return InlineKeyboardMarkup(keyboard)

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    job_queue = app.job_queue
    
    now = datetime.now(pytz.utc)
    
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    initial_delay_1min = (next_minute - now).total_seconds()
    
    next_30sec = now.replace(second=30 if now.second < 30 else 0) + timedelta(minutes=0 if now.second < 30 else 1)
    initial_delay_30sec = (next_30sec - now).total_seconds()
    
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
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("history", show_history))
    app.add_handler(CommandHandler("ek", ek_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url_input))
    
    logging.info("Multi-Platform Lottery Predictor is running...")
    app.run_polling()

if __name__ == '__main__':
    main()
