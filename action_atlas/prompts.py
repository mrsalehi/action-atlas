GPT4_GET_DESCRIPTIONS="""I give you five action names in sport {sport} and you have to describe that action in two to three sentences. Try to be concise but accurate and cover the important and unique characteristics of each action. Separate the descriptions with two new lines. Don't write the action number and name before the description just description separated by two new lines.\n{options}"""

GPT4_GET_DIFFERENCE_WITH_HARD_NEGATIVES="""I give you an action name in a sport and 9 hard negatives for it. You have to write the difference between the action and the hard negatives.
The difference should be no more than two to three sentences. Focus on the very key elements that differentiate the action from the hard negatives. I'm not an expert in sports so explain it in a way that a non-expert can understand.
---
EXAMPLE:
SPORT: Soccer
ACTION: Double stepover
HARD_NEGATIVES: Cruyff Turn, Elastico, Roulette, Rainbow Flick, Nutmeg, Drag Back, Rabona, Maradona Turn, Heel Flick
RESPONSE:
Cruyff Turn:
The Double Stepover involves a player rapidly spinning their foot around the ball twice without touching it to mislead an opponent. In contrast, the Cruyff Turn is performed by feinting a pass or shot before pulling the ball behind the standing leg with the inside of the foot, often to turn away from the defender.

Elastico:
The Double Stepover is a trick primarily used to confuse defenders with quick, circular foot movements around the ball. The Elastico, however, involves pushing the ball in one direction with the outside of the foot and then quickly snapping it back in the opposite direction with the inside of the same foot.

Roulette:
The Double Stepover is performed in front of the ball to feint direction changes. The Roulette, or Zidane turn, involves a player using one foot to drag the ball back and the other to spin 360 degrees around it, effectively shielding the ball with their body.

Rainbow Flick:
The Double Stepover does not involve contact with the ball until the move is completed. In contrast, the Rainbow Flick entails flicking the ball up from behind with one foot and then using the other foot to lift it over both the player and the opponent, usually when the opponent is approaching from behind.

Nutmeg:
The Double Stepover is used to create space and mislead defenders through body feints around the ball. A Nutmeg, however, specifically aims to pass the ball between an opponent’s legs, exploiting their open stance.

Drag Back:
While the Double Stepover involves moving around the ball to deceive, the Drag Back is a simpler maneuver where the ball is pulled back with the sole of the foot to change direction quickly or retain possession under pressure.

Rabona:
The Double Stepover is a feint performed with natural foot movements around the ball. The Rabona is a flamboyant technique where a player wraps one leg behind the standing leg to strike the ball, often used for crossing or shooting from an awkward position.

Maradona Turn:
The Double Stepover involves deceptive footwork without body rotation around the ball. The Maradona Turn, similar to the Roulette, combines a drag back with a 360-degree turn, using the body extensively to shield the ball.

Heel Flick:
The Double Stepover is about misleading defenders with foot feints around the ball. The Heel Flick, however, involves flicking the ball backward with the heel, often to pass to a teammate or to clear space when running forward.
---
SPORT: {sport}
ACTION: {action}
HARD_NEGATIVES: {hard_negatives}
RESPONSE:
"""

MC_PROMPT ="""Answer the given question according to the video presented as a sequence of frames. Only output the choice number and nothing else. When answering the question consider all legal and illegal moves and drills.\n{question}\n{options}"""

MC_WITH_DESCRIPTION ="""Answer the given question according to the video presented as a sequence of frames. Each choice presents an action name along with its description which helps in identification. Only output the choice number and nothing else. When answering the question consider all legal and illegal moves and drills.\n{question}\n{options}"""

MC_WITH_DESCRIPTION_COT ="""Answer the given question according to the video presented as a sequence of frames. Each choice presents an action name along with its description which helps in identification. You can describe the video or do any step by step reasoning about what you see in the video and the choices. However, output your final choice number (1 to 5) at the end of your response in a new line. When answering the question consider all legal and illegal moves and drills.\n{question}\n{options}"""

MC_COT = """Answer the given question according to the video. You can describe the video or do any step by step reasoning about what you see in the video. However, output your final choice number (1 to 5) at the end of your response in a new line.\n{question}\n{options}"""

MC_STEP_BY_STEP_ACROSS_FRAMES="""Answer the given question according to the video presented as a sequence of frames. Each chioce presents an action name along with its description which helps in identification. You must do step by step reasoning across frames and describe the changes that happen between each two consecutive frames. You can also do step by step reasoning about options and why they might be correct or wrong. However, output your final choice number (1 to 5) at the end of your response in a new line.\n{question}\n{options}. Let's think step by step across frames:"""

GPT4_WRITE_HARD_NEGATIVES="""Write some hard negatives for move {action} in sport {domain}.
The negatives should be plausible and EXTREMELY hard to distinguish from the correct answer. However, THEY MUST BE WRONG AND DIFFERENT from the correct one. Also, the hard negatives must be well-known {domain} moves.
for each action, write 9 hard negatives. and write one hard negative in a line without any bullet points or numbers.
----
EXAMPLE:
ACTION: windshield wiper forehand
VERY HARD NEGATIVES:
Inside-out forehand
Topspin lob
Slice backhand
Flat serve
Kick serve
Reverse forehand
Volley at the net
Drop shot
Two-handed backhand
----
ACTION: {action}
VERY HARD NEGATIVES:
"""

GPT4_WRITE_QUESTION_VIDEO_BENCHMARK="""I will give you some information about a sport video, and you should generate a question based on the info.
The information:
1. an action.
1. description of the person performing the action.
3. what happens before the action.
4. what happens after the action.
Note that 3 and 4 could be "none". If both are "none", then just focus on the action and the person performing the action.
NOTE THAT YOU MUST NOT REFER TO THE ACTION NAME IN YOUR QUESTION!!
---
Example1:
ACTION: alley-oop dunk
PLAYER: player number 34 wearing white jersey
BEFORE: player number 34 runs towards the basket
AFTER: none
QUESTION: What best describes the move made by player wearing white jersey with number 34 after they run towards the basket?
---
Example2:
ACTION: Throwing
PLAYER: the man in the camo shirt and black pants
BEFORE: none
AFTER: none
QUESTION: What best describes the action that the man in the camo shirt and black pants performs?
---
Example3:
ACTION: Hedge
PLAYER: It is a man with a white headband and the number 34 on his jersey
BEFORE: He was guarding number 32 on the opponent team
AFTER: The opponent loses the ball
QUESTION: What best describes the action that the man with a white headband and the number 34 on his jersey performs after he was guarding number 32 on the opponent team and before the opponent loses the ball?
---
Example
ACTION: {action}
PLAYER: {player}
BEFORE: {before}
AFTER: {after}
QUESTION: """

GPT4_FILTER_OUT_NON_PHYSICAL_ACTIONS_FEW_SHOT_PROMPT = """I would give you a list of possible actions in {sport}. Your task is to specify which one of them are PHYSICAL actions that rquire MOVEMENT that can be captured in a video. Also the action has to be specific and not a general term in that sport.
Here are some examples for the kinds of actions I am looking for in a few example sports:
Alley-oop dunk in basketball
Around the world in soccer
Cross in soccer
Cruyff turn in soccer
Offensive rebound in basketball
Panenka in soccer
---
I give you 10 possible actions in {sport} and only write the name of those that are physical with movement in separate lines. Only output the exact name of actions nothing else. If none of the actions met the criteria output "".
{actions}
"""

GPT4_FILTER_OUT_NON_PHYSICAL_ACTIONS_PROMPT = """Give a short one sentence description about {action} in {sport}. Given the description, is it a physical action that requires movement that can be captured in a video? Write the description in first line and answer to the question with just "yes" or "no" in second line"""

GPT4_FILTER_OUT_NON_ACTIONS_PROMPT = """I give you a phrase that is relevant to the sport {sport}. You have to give 1.one sentence description of the phrase in one line and 2. if it's a major ACTION or MOVE in {sport} and is grounded and can be captured in a youtube video in the next line using words "yes" and "no".
Just output the sentence in first line and "yes" or "no" in the next line and nothing else:
{action} in sport {sport}
"""

GPT4_EXPAND_INITIAL_ACTION_LIST="""I give you an initial list of actions in {sport}. YOU HAVE TO EXPAND THIS LIST AND GIVE A COMPREHENSIVE LIST OF ALL KNOWN ACTIONS, SHOTS, MOVES, ETC. IN THIS SPORT. 
It's crucial that you include all the well known actions, shots, and moves specially those that might have a wikipedia page. Give a list without description, without bullets and numbers, and just the action names line by line. Also do not include multiple actions inside parentheses. List these kind of actions as separate items.
Here is my list, rewrite and expand it:
{actions}
"""

GPT4_FILTER_PARTIAL_MATCH="""Here is the title of a youtube video about sport {sport}.Answer two questions:
1. Based on the title, is the video about {action}? Only answer "yes" or "no".
2. If the title is not about {action}, then what is the closest action to the title? If there is no close action, then answer "none". also if the title is about multiple actions, then write all their names separated by commas.
"""


GPT4_EXPAND_EXPANDED_ACTION_LIST="""I give you an initial list of actions in {sport}. YOU HAVE TO EXPAND THIS LIST AND GIVE A COMPREHENSIVE LIST OF ALL KNOWN ACTIONS, SHOTS, MOVES, ETC. IN THIS SPORT. 
It's crucial that you include all the well known actions, shots, and moves specially those that might have a wikipedia page. Rules:
1. Give a list without description, without bullets and numbers, and just the action names line by line.
2. Optimize the list for youtube search, so don't make the action name too long.
3. do not use paranthese, or slashes in your lines. For instance, if the action has multiple names such as "Standard Shot (Instep Drive)" then write "Standard Shot" and "Instep Drive" in two separate lines. Also do not write more description about an action in parantheses, just the action name.
4. Do not categorize the actions, just give a simple plain list of action names nothing else.
Here is my list, rewrite and expand it:
{actions}
"""

FIND_ACTIONS_GIVEN_SPORT="""I'll give you a sport name and you have to generate a list of actions that are commonly associated with that sport.
1. Please only list actions that are well-known but the list should be as exhaustive as possible.
2. If an action has multiple types list all of them. For instance in soccer there are different types of shoots such as Standard Shot (Instep Drive), Chip Shot, Curve Shot, Knuckleball Shot etc. Output all of them and each type should be in a new line.
3. If an action has two names, separate them with a slash. For instance in soccer, "Standard Shot / Instep Drive".
EXAMPLE:
---
SPORT: golf
RESPONSE:
Drive/tee shot
Fairway shot
Approach shot
Chip shot
Putt
Bunker shot
Pitch shot
Flop shot
Punch shot
Recovery shot
---
SPORT: {sport}
RESPONSE:\n
"""

FIND_SPORTS_PROMPT="""
give an exhaustive list of common sports that you know about. Just list them with numbers and it should contain more than 100 sports.
"""

LOCALIZE_ACTION_PROMPT="""This is a youtube video about {action} in {sport}. 
Title of the video: {title}. I'm providing the transcribed speech from the video with its time segments. the first number in each line is the start time and the rest is the subtitle.
Your task is to localize the time segments where YOU ARE ABSOLUTELY CERTAIN that a {action} is happening in the video.
Here are some clues to use:
1. If based on title and speech, it's a sports commentary, then look for places where the commentator mention something about the action done well. For instance the commentator might say "that was a great {action}".
2. If based on title and speech, it's a tutorial, then look for places where the speaker specifies "Now let's do {action}" or something that implies that. For instance they can say "let's practice it once again". Note that in such videos, usually in the beginning, the speaker might explain about the action and its techniques but the actual action
might happen later". Usually when the verb tense is future, the action is not happening at that time so ignore such sentences.
3. Only output the time-stamps. If it happens at 10, 150, and 240 then output 10 240 150. If no action is happening, output -1.
---
{speech}---
"""

GPT4_ANNOTATOR_DESCRIPTION = """I want to give a video to an annotator on Amazon mechanical turk and the annotator should find when the action "{action}" in sport {sport} happens in the video. 
Assume that they don't know much about {sport}. Explain what {action} is and what they should look for in the video.
Here is an example for action "step over" in Soccer to help you learn the output template that I want.
---
Explanation for Amazon Mechanical Turk Annotators:
Task Description:
Your task is to watch a soccer (football) video and identify moments when a player performs a move known as a "step over." It is important to watch the players' feet and the ball closely to accurately spot these actions.

What is a "Step Over"?
A "step over" is a dribbling move used by soccer players to trick an opponent into thinking they are going to move the ball in a certain direction, but instead, they move it in the opposite direction or retain its current path. This move is often used to outmaneuver a defender and create space for advancing towards the goal or making a pass.

Key Elements to Look For:
Player with the Ball: Focus on players who are in possession of the ball and are facing an opponent directly, usually a defender.

Foot Movement: The player will quickly move one foot around the front of the ball (without touching it) in a circular motion. It looks as if they are going to kick or push the ball in the direction their foot is moving, but they don't.

Body Feint: Along with the foot movement, the player might also use their body to fake a move in a certain direction. They may lean or look one way to further deceive the defender.

Continuation of Play: After performing the step over, the player either continues to dribble the ball in the original direction or changes direction swiftly, bypassing the opponent. This is the critical moment of the step over, where the trickery culminates in the player's next move.

Frequency: This move can happen multiple times and by different players throughout the game. It is a common dribbling technique.

Examples and Practice:
Practice Spotting: It might be helpful to watch a few examples of "step overs" in soccer training videos or match highlights to get a sense of the move before starting the task.
Video Speed: Consider slowing down the video playback speed if you're having trouble spotting the move. This can make it easier to see the quick footwork involved in a step over.
Your Task:
As you watch the video, note the timestamp(s) when you believe a "step over" occurs. If possible, describe the action and the player using visual cues, such as the number on their back, their name if visible (e.g., "blue team player near the corner flag at 02:15") to provide context for your observations.

Remember, this task requires attention to detail, as the move can be quick and subtle. Your accurate identification of these moments is valuable for our analysis.
---
"""

GPT4_ANNOTATOR_DESCRIPTION_V2 = """I want to give a video to an annotator on Amazon mechanical turk and they should find when the action "{action}" in sport "{sport}" happens in the video.
Assume that they don't know about {sport} at all. Explain what {action} is in simple terms and what they should look for in the video. Be comprehensive but explain everything in simple terms.
Here is an example for "jab step" in Basketball to help you learn the output template. Just use this as a template so don't use any basketball related info from this example in your response.
---
In basketball, a jab step is a quick, deceptive movement used by the player with the ball to test the defender’s reaction or to create space for a shot or drive. It involves the player with the ball making a short, sharp step with one foot while keeping the other foot planted as a pivot foot. The movement looks like a quick poke or "jab" toward the floor, which is where the name comes from.
Simple Steps to Identify a Jab Step:

1. Identify the Player with the Ball: First, focus on the player who is holding the basketball. This is important because only the player with the ball can perform a jab step.

2. Look at Their Feet: Observe the player’s feet closely. One foot will remain in contact with the floor at all times; this is the pivot foot. The other foot will quickly step forward, to the side, or even backward.

3. Watch the Quick Step: The key to spotting a jab step is to see a quick, short step by the non-pivot foot. This step is not meant to move the player significantly in any direction; instead, it’s meant to make the defender think they are going to move.

4. Check the Defender’s Reaction: Often, the purpose of a jab step is to see how the defender reacts. If the defender moves or flinches backward or to one side, the jab step has done its job.

5. Look for Subsequent Actions: After the jab step, watch what the player does next. They might shoot, pass, or drive towards the basket depending on the defender’s reaction.

Note the Timing: Whenever you observe these elements—the player with the ball, the quick step with one foot, and the defender's reaction—note the time in the video. This is likely when a jab step is happening.
---
"""