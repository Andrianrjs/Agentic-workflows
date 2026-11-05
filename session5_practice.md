# Session 5 - Practice

Today's goal is to adapt our Chainlit app so that it logs relevant data to Langfuse, and work in Langfuse's UI to add relevant visualizations and metrics.

The code structure in this practice won't change!

# 1. Logging to Langfuse

The objective of this section is to adapt our Chainlit app to log relevant data to Langfuse.

This means logging at least:
- The raw user input
- The input sent to the model (parameters and templated prompt)
- The raw model output and metadata (model name, version, latency)
- The extracted response shown to the user

## Langfuse setup

1. Follow this [tutorial](https://langfuse.com/self-hosting/deployment/docker-compose) to self-host Langfuse using Docker Compose.
2. Log into the Langfuse UI at `http://localhost:3000` and create an organization and a project.
3. In the project settings, create a new API key. Use them to add the following environment variables to your `.env` file:
    ```env
    LANGFUSE_SECRET_KEY=sk-...
    LANGFUSE_PUBLIC_KEY=pk-...
    LANGFUSE_HOST=http://localhost:3000/
    ```
4. Add a langfuse import at the top of your Chainlit application file, and run your Chainlit app and send a message!
    ```python
    from langfuse import Langfuse

    langfuse = Langfuse()
    ```
5. You should see some traces appear in the Langfuse UI, but where do these come from?

## Adapting the Chainlit app

1. Chainlit generates Open Telemetry traces by default, but these aren't the traces that interest us the most. To remove them from Langfuse's scope, block the `chainlit` instrumentation scope when initializing Langfuse:
    ```python
    langfuse = Langfuse(
        blocked_instrumentation_scopes=["chainlit"]
    )
    ```
2. Now, we need to add custom logging to our Chainlit app. Wrap the function that calls the model API with an `observe` decorator, and send a message to your Chainlit app to see what appears in Langfuse.
    - Does this contain all the relevant metadata you want to log?

Let's now split the logging into two parts:
- A `span` that has as input the raw user message, and as output the extracted response shown to the user.
- A nested `generation` that has as input the prompt sent to the model (with parameters), and as output the raw model response and metadata.

To implement this, follow these steps:
1. The `observe` decorator will create the first span. You can update attributes of the span using the `langfuse.update_current_span` method. For example:
    ```python
    langfuse.update_current_span(input=message.content)
    ```
2. In the model call function, create a `generation` using the `langfuse.start_as_current_generation` context manager. For example:
    ```python
    with langfuse.start_as_current_generation() as gen:
        
        # Call the model API
        gen.update(
            input=templated_input,
            ... # other attributes to update
        )
    ```
3. Re-run your chainlit app and send a message. Does the trace look better now? Feel free to iterate until all relevant data is logged correctly.

# 2. Implementing metrics

Once you log traces correctly, you can start implementing custom online metrics in Langfuse. 

These run on each trace as it arrives, and can be used to create dashboards and alerts for live monitoring!

## Setting up an LLM evaluator

Most metrics are LLM-as-a-judge evaluations - this means you need to set up an LLM evaluator in Langfuse first.

1. Go to your project settings, and create a new LLM connection using a Gemini or OpenAI API key.
2. Go to the "LLM-as-a-judge" tab in the Langfuse UI, and create a new evaluator.
3. In the upper right corner, you can define a default model to use as evaluator. You should find models available given the LLM connection you set up - choose one of them!

## Adding a custom LLM as a judge metric

Let's create a Yoda-ness critic metric.

1. With your default evaluator set up, go to the "LLM-as-a-judge" tab and select "Set up evaluator".
2. Take a look at some of the built-in evaluators, and their prompt templates. Take a look at the templating logic in particular!
    - As you can see, `ragas` is integrated in Langfuse!
3. Select "Create custom evaluator", and fill in your prompt. A good practice is to provide a couple examples (few-shot prompting).
    - If you wish to refer to fields in the trace, you can use the double bracket syntax (for example, `{{input}}`).
4. Then, return to the "LLM-as-a-judge" tab, and click the "Set up evaluator" button again - this time, your new custom evaluator should appear in the list.
5. In the following screen you can decide if you want the evaluator to run on all new traces, and also on previously existing traces. You can also add filters if necessary.
6. After saving, you should start seeing new metrics appear in your traces (in the "Scores" tab, the "Tracing" view, or the "LLM-as-a-judge" tab).
7. Take a look at a couple examples. Is your metric working as expected?

>Note: Langfuse also comes with interfaces for human annotation, and dataset creation. Feel free to explore these features on your own!

# 3. Creating a custom dashboard

Once you have metrics set up, you can create custom dashboards to monitor them.

1. Go to the "Dashboards" tab in the Langfuse UI, and create a new dashboard.
2. Add a new widget to your dashboard that tracks some of the metrics you created over time. 

You should be setup for live monitoring of your Chainlit app now, congratulations for having completed this practice, and the entire course! ✨
