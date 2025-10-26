# Prolific Setup Guide

This guide walks you through setting up Prolific for collecting human preferences in your RLHF pipeline.

## 1. Create a Prolific Account

1. Go to [prolific.com](https://www.prolific.com/)
2. Sign up for a Researcher account (collect data)
3. Complete your profile and organization details
4. Add funds to your account (minimum £5.00)

## 2. Generate an API Token

1. Log in to your Prolific account
2. Navigate to **API Tokens**
3. Click **"Create New Token"**
4. Give your token a descriptive name (e.g., "RLHF Pipeline")
5. Copy the token
6. Save it in your `.env` file:
   ```bash
   PROLIFIC_API_TOKEN=your_token_here
   ```

## 3. Find Your Workspace ID

1. In Prolific, click on your workspace name
2. The workspace ID is in the URL: `https://app.prolific.com/researcher/workspaces/{WORKSPACE_ID}`
3. Copy this ID to your `.env` file:
   ```bash
   PROLIFIC_WORKSPACE_ID=your_workspace_id_here
   ```
4. The same process can be applied for a project ID

## 4. Study Design Best Practices

### Participant Compensation

The minimum payment on Prolific is **at least £6.00/$8.00 per hour**. This is an absolute minimum, and researchers are strongly recommended to pay at least **£9.00/$12.00 per hour** to ensure fair compensation and data quality. Payments are calculated based on the estimated completion time and the reward amount.


### Eligibility Requirements

Prolific gives you full control over who participates in your study, allowing you to target, include, or exclude participants based on a wide range of attributes and qualifications.

1. ** Screen from Hundreds of Participant Attributes**: You can filter any and all participants based on demographics (e.g., age, gender, location, lifestyle) and hundreds of other characteristics already collected by Prolific.
2. **Create Custom Screeners**: Define your own custom eligibility criteria using bespoke questions or logic to capture specific traits or experiences not covered by built-in prescreeners.
3. **Use AI Taskers**: Access a curated group of pre-qualified participants specifically skilled at model evaluation, testing, and alignment. AI Taskers excel in reading comprehension, writing, systematic reasoning, and complex multi-step instructions, ensuring reliable, high-quality feedback for your RLHF or model evaluation pipeline. These taskers are ideal for:
- Model evaluation and benchmarking
- Side-by-side performance comparisons
- Factuality and accuracy assessments
- Generative and creative tasks
4. **Use Domain Experts**: Get detailed, expert-level feedback from verified professionals with deep subject matter expertise. Domain Experts are verified via assessments, academic credentials (e.g., Google Scholar), professional registration (e.g., medical databases), and industry experience (e.g., LinkedIn verification). Perfect for:
- Technical accuracy validation
- Domain-specific training data collection
- Professional content evaluation
5. **Include or Exclude Specific Participants**: Avoid sampling bias by excluding participants who have taken similar studies or by building custom allowlists and blocklists. You can also include trusted participants who’ve previously delivered high-quality work, ensuring consistency in your dataset.

### Instructions for Participants

Provide clear, concise instructions:

```markdown
# Task Instructions

In this study, you will compare pairs of AI-generated responses to questions.

## What to do:
1. Read the prompt/question
2. Read both Response A and Response B
3. Select which response is **better overall**

## Evaluation Criteria:
Consider which response is:
- ✓ More accurate and factually correct
- ✓ More helpful and complete
- ✓ Clearer and better organized
- ✓ More appropriate and safe

## Important:
- There are no "right" or "wrong" answers
- We want your genuine judgment
- If both are equal, choose the one you slightly prefer
- Don't rush - quality matters

## Time Required:
Approximately 10 minutes for 20 comparisons
```

## 5. Testing Your Study

### Before launching:

1. **Preview Mode**: Test with 1-2 participants
2. **Pilot Run**: Launch with 10 participants
3. **Check Quality**: Review initial responses
4. **Adjust**: Modify instructions or pay if needed
5. **Full Launch**: Scale to full participant count

### Handle Issues

**Low response rate?**
- Increase reward
- Adjust eligibility requirements
- Improve study description

**Quality issues?**
- Add more attention checks
- Increase minimum approval rate
- Add screening questions

## Resources

- [Prolific API Documentation](https://docs.prolific.com/docs/api-docs/public/)
- [Prolific Researcher Help Center](https://researcher-help.prolific.com/)
