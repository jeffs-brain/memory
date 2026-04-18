// SPDX-License-Identifier: Apache-2.0

package lme

import "fmt"

// JudgePromptVersion is bumped on any change to the judge prompt or
// response schema. Runs with different prompt versions are not directly
// comparable.
//
// v6: the judge call is wrapped in [completeJSON] which installs a
// `{verdict, rationale}` JSON schema response format.
const JudgePromptVersion = 6

// The official LongMemEval evaluation prompts from the paper's codebase
// (xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py). Each category
// has its own prompt. The judge returns binary yes/no.

const judgePromptStandard = `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: %s

Correct Answer: %s

Model Response: %s

Is the model response correct? Answer yes or no only.`

const judgePromptTemporal = `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: %s

Correct Answer: %s

Model Response: %s

Is the model response correct? Answer yes or no only.`

const judgePromptKnowledgeUpdate = `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: %s

Correct Answer: %s

Model Response: %s

Is the model response correct? Answer yes or no only.`

const judgePromptPreference = `I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: %s

Rubric: %s

Model Response: %s

Is the model response correct? Answer yes or no only.`

const judgePromptAbstention = `I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: %s

Explanation: %s

Model Response: %s

Does the model correctly identify the question as unanswerable? Answer yes or no only.`

// judgePromptForCategory returns the correct official LME evaluation
// prompt for the given question category.
func judgePromptForCategory(category string, isAbstention bool) string {
	if isAbstention {
		return judgePromptAbstention
	}
	switch category {
	case "temporal-reasoning":
		return judgePromptTemporal
	case "knowledge-update":
		return judgePromptKnowledgeUpdate
	case "single-session-preference":
		return judgePromptPreference
	default:
		return judgePromptStandard
	}
}

// formatJudgePrompt renders the category-appropriate judge prompt and
// prepends a "Question date:" anchor when one is known.
func formatJudgePrompt(category string, isAbstention bool, question, groundTruth, response, questionDate string) string {
	template := judgePromptForCategory(category, isAbstention)
	body := fmt.Sprintf(template, question, groundTruth, response)
	if questionDate == "" {
		return body
	}
	return fmt.Sprintf("Question date: %s\n\n%s", questionDate, body)
}
