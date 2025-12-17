#!/bin/bash

# Script to send sample feedback items for retraining
# Usage: ./send_feedback.sh [count]
# Default: 15 feedback items

API_URL="http://ml-sentiment-alb-850542821.us-east-2.elb.amazonaws.com"
COUNT=${1:-15}

echo "Sending $COUNT feedback items to $API_URL/feedback"
echo "=================================================="

# Positive feedback samples
positive_texts=(
    "This product is absolutely amazing, best purchase ever"
    "Love it, works perfectly and exceeded my expectations"
    "Excellent quality, highly recommend to everyone"
    "Great value for money, very satisfied with this purchase"
    "Fantastic product, will definitely buy again"
    "Outstanding quality and fast shipping"
    "Perfect, exactly what I was looking for"
    "Best product I have ever bought, 5 stars"
)

# Negative feedback samples
negative_texts=(
    "Terrible product, broke after one day of use"
    "Waste of money, do not buy this"
    "Very disappointed, poor quality materials"
    "Does not work as described, returning it"
    "Horrible experience, worst purchase ever"
    "Cheaply made, fell apart immediately"
    "Complete garbage, want my money back"
)

# Neutral feedback samples
neutral_texts=(
    "Product arrived on time, works as expected"
    "Its okay, nothing special about it"
    "Average product, does the job"
    "Decent quality for the price"
    "Not bad, not great either"
)

sent=0

# Send positive feedback
for text in "${positive_texts[@]}"; do
    if [ $sent -ge $COUNT ]; then break; fi
    echo -n "[$((sent+1))/$COUNT] Sending positive: "
    response=$(curl -s -X POST "$API_URL/feedback" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\", \"model_prediction\": \"positive\", \"user_label\": \"positive\"}")
    echo "$response"
    ((sent++))
    sleep 0.5
done

# Send negative feedback
for text in "${negative_texts[@]}"; do
    if [ $sent -ge $COUNT ]; then break; fi
    echo -n "[$((sent+1))/$COUNT] Sending negative: "
    response=$(curl -s -X POST "$API_URL/feedback" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\", \"model_prediction\": \"negative\", \"user_label\": \"negative\"}")
    echo "$response"
    ((sent++))
    sleep 0.5
done

# Send neutral feedback
for text in "${neutral_texts[@]}"; do
    if [ $sent -ge $COUNT ]; then break; fi
    echo -n "[$((sent+1))/$COUNT] Sending neutral: "
    response=$(curl -s -X POST "$API_URL/feedback" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\", \"model_prediction\": \"neutral\", \"user_label\": \"neutral\"}")
    echo "$response"
    ((sent++))
    sleep 0.5
done

echo ""
echo "=================================================="
echo "Sent $sent feedback items"
echo ""
echo "To trigger retraining, run:"
echo "curl -X POST $API_URL/retrain"
