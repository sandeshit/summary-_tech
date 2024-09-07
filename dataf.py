import pandas as pd

# Data (list of dictionaries with Date and Message only)
data = [
    {"Date": "", "Message": "I'll be working remotely as I moved to Imadol last night. I'm not feeling well, so I'll be taking breaks to rest throughout the day."},
    {"Date": "", "Message": "Been feeling sick since yesterday. I'll be resting for the first half of the day. I'll join office in the second half."},
    {"Date": "", "Message": "Since nobody will be present at home tomorrow. I will be working from home tomorrow."},
    {"Date": "", "Message": "I will be late for standup or on first half leave tomorrow depending on how quickly the car issues will be resolved at the service center."},
    {"Date": "", "Message": "I will be little bit late. Forgot phone at room while coming to office, got back to room again."},
    {"Date": "", "Message": "I'm working from home today as well!"},
    {"Date": "", "Message": "I have not been feeling well since yesterday evening (headache and dizziness again) so I will be working lightly remotely for today at least with rests. As for tomorrow, I will keep you guys updated depending on how I feel. Take care, guys."},
    {"Date": "", "Message": "Waiting for my turn for defense might be a bit late for standup."},
    {"Date": "", "Message": "I will be out for a while, I will be having lunch outside. See you after lunch."},
    {"Date": "", "Message": "Will be on leave for the first half. Nobody at home and ama not feeling well."},
    {"Date": "", "Message": "I have tonsils. Body needs a bit of a rest. I will be working from home today."},
    {"Date": "", "Message": "I'm experiencing throat pain and a severe headache from last night, so I'll be working remotely today."},
    {"Date": "", "Message": "I'll be on first half leave. Need to visit government offices for some documents. Best of luck to me. I'll join in the second half."},
    {"Date": "", "Message": "I will on leave today. Not feeling well."},
    {"Date": "", "Message": "I am at college. I will be 1 hour late for office."},
    {"Date": "", "Message": "I will be leaving about 60 to 90 minutes earlier today. And I am on leave tomorrow."},
    {"Date": "", "Message": "Period cramps, I'll be on leave today."},
    {"Date": "", "Message": "There is a small Janmastami puja at home, I will be on first half leave. Will join office in the second half."},
    {"Date": "", "Message": "I'll be working from home for a couple of days. Sechen is alone and I need to be on standby for Puntu. I won't be able to join the stand-up today."},
    {"Date": "", "Message": "I am leaving early today for some work. Have a nice weekend."},
    {"Date": "", "Message": "I will be working from home for the first half and will be on leave for the second half of the day as I have to take Mamu to the hospital for some tests."},
    {"Date": "", "Message": "As mentioned in the stand-up yesterday, I will be working first half from home and then taking a second half leave to attend a friend's wedding reception. Have a good weekend, everyone. Cheers!"},
    {"Date": "", "Message": "I'm dealing with severe back pain, so I'll be working from home. I might also step out for a hospital follow-up."},
    {"Date": "", "Message": "I might be little late for standup."},
    {"Date": "", "Message": "I might be a bit late due to a huge traffic jam. I might miss standup!"},
    {"Date": "", "Message": "I will be on first half leave. Will join office on second half of the day."},
    {"Date": "", "Message": "As I mentioned yesterday, will be working from home today. The pooja is going on so I will have to miss the standup."},
    {"Date": "", "Message": "I will be on leave today."}
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the DataFrame

df = df.drop('Date',axis = 1)

print(df)