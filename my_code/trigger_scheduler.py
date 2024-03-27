from clearml.automation import TriggerScheduler

# create the TriggerScheduler object (checking system state every minute)
trigger = TriggerScheduler(pooling_frequency_minutes=1.0)

# Add trigger on dataset creation
trigger.add_dataset_trigger(
    name='Retrain On Dataset',
    # schedule_function=lambda x: print("Hey Mom!"),
    schedule_task_id='9a9d133722264fda805677de08055cf7', # you can also schedule an existing task to be executed
    schedule_queue='default',
    trigger_project='My First Project',
    trigger_name='Models' # Dataset Name
)

trigger.start() #  Replace with trigger.start_remotely() to start remotely
