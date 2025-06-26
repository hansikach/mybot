from openai import OpenAI

from util.log_proc_time import log_time


@log_time
def get_llm_response(prompt):
    client = OpenAI(base_url="http://localhost:12434/engines/v1", api_key="ignored")

    resp = client.completions.create(
        model="ai/gemma3",
        prompt=prompt,
        max_tokens=512
    )

    print(resp.choices[0].text)

get_llm_response("I am running the gemma3 llm on docker using docker model-runner. Can I optimize the response time in any ways? If yes, how? I mean can I keep the modal loaded ready so that it would not load for each API call and inference quickly? If yes, how? Please provide the code to do so.")