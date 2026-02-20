# Phone Agent Test Script

Call the Twilio number and follow these test scenarios in order.
After each step, note PASS/FAIL and any observations.

---

## 1. Greeting (Auto)

**Expected:** Agent greets you as Thalia from Deepgram.

---

## 2. Knowledge Base — Basic Question

**Say:** "What is Deepgram?"

**Expected:**
- You hear a filler like "Let me look that up" almost immediately
- Agent responds with accurate info about Deepgram (speech-to-text, text-to-speech, voice AI)

---

## 3. Knowledge Base — Specific Question

**Say:** "What languages does Deepgram support?"

**Expected:**
- Filler plays quickly
- Agent responds with supported languages from the knowledge base

---

## 4. Knowledge Base — Technical Question

**Say:** "How do I use the Deepgram streaming API?"

**Expected:**
- Filler plays quickly
- Agent responds with technical details about the streaming API

---

## 5. Voice Switch — Male

**Say:** "Can you switch to a male voice?"

**Expected:**
- Agent responds in a MALE voice (Orion or similar)
- Agent introduces itself with the new name

---

## 6. Voice Switch — Female

**Say:** "Switch to a female voice please."

**Expected:**
- Agent responds in a FEMALE voice
- Agent introduces itself with the new name

---

## 7. Accent — British Male

**Say:** "Can you speak with a British male accent?"

**Expected:**
- Agent switches to Draco (British male)
- Agent introduces itself as Draco

---

## 8. Accent — Australian Female

**Say:** "Switch to a female Australian voice."

**Expected:**
- Agent switches to Theia (Australian female)
- Agent introduces itself as Theia

---

## 9. Accent — American Female

**Say:** "Go back to an American female voice."

**Expected:**
- Agent switches to Thalia (American female)
- Agent introduces itself as Thalia

---

## 10. Question After Voice Switch

**Say:** "What is Deepgram's text to speech product?"

**Expected:**
- Filler plays in the CURRENT voice
- Agent answers in the CURRENT voice (not the original)

---

## 11. Barge-In Test

**Wait for agent to start a long response, then interrupt:**

**Say:** (interrupt mid-sentence) "Stop. What pricing plans does Deepgram offer?"

**Expected:**
- Agent stops speaking immediately
- Agent processes the new question

---

## 12. Language — Spanish

**Say:** "Can you switch to Spanish?"

**Expected:**
- Agent switches to a Spanish voice (Estrella or Nestor)
- Agent introduces itself in Spanish

---

## 13. Language — French

**Say:** "Switch to French please."

**Expected:**
- Agent switches to a French voice (Agathe or Hector)
- Agent introduces itself in French

---

## 14. Language — Back to English

**Say:** "Switch back to English please."

**Expected:**
- Agent switches back to an English voice
- Agent responds in English

---

## 15. End Call — Polite Goodbye

**Say:** "Thank you, goodbye!"

**Expected:**
- Agent says a farewell message like "Thank you for calling! Have a great day!"
- Call ends gracefully

---

## Quick Reference — Available Voices

| Accent     | Female    | Male      |
|------------|-----------|-----------|
| American   | Thalia    | Orion     |
| British    | Pandora   | Draco     |
| Australian | Theia     | Hyperion  |
| Spanish    | Estrella  | Nestor    |
| French     | Agathe    | Hector    |
| German     | Viktoria  | Julius    |
| Italian    | Livia     | Dionisio  |
| Dutch      | Rhea      | Sander    |
| Japanese   | Izanami   | Fujin     |
