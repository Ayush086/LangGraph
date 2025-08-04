# Persistance
It helps to store the history of tasks performed and also to retrieve the state of workflow over time. Stores both final output as well as the intermidiate output (at checkpoints).

## Components
### 1. Checkpointers
- Point at which state of data will be stored.  
- They are present at each super step (steps other than initial step).

### 2. Threads
- Used to execute multiple workflows simultaneously without interferring with each other by keeping their state to workflow specific only.
- ```thread_id``` is used to identify the workflow specific states

### Usecases
**1. Short Term Memory**
- helps in saving the previous history.
- If we re-continue the chat/perform any task it will do it without loosing any previous data.

**2. Fault Tolerance**
  - Even if system crashes it restart the process from the state where it broke not from the initial state.

**3. Human In The Loop (HITL)**
  - If generated output is approved by human then only further steps will be performed else it will re-iterate the process till it get's the approval.
 
**4. Time Travel**
  - We can access any intermidiate state and modify the it and can generate new final output based on the modified changes.
  - Replay the workflow from any checkpoint.
