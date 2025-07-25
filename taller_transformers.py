from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
model_name = "distilgpt2"  # A lightweight pretrained transformer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_next_token_probability(prompt: str, next_token_str: str):
    """Compute probability of the next token given a prompt."""
    # Tokenize prompt and next token
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    next_token_id = tokenizer.encode(next_token_str, add_special_tokens=False)
    
    # Get logits for next token prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)
    
    # Take logits for the last token in prompt
    last_token_logits = logits[0, -1, :]
    probs = torch.softmax(last_token_logits, dim=0)
    
    # For multi-token next_token_str, calculate joint prob by chain rule
    prob = 1.0
    cur_input_ids = inputs.input_ids
    for token_id in next_token_id:
        with torch.no_grad():
            outputs = model(input_ids=cur_input_ids)
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            last_probs = torch.softmax(last_logits, dim=0)
            token_prob = last_probs[token_id].item()
            prob *= token_prob
            # Append token_id for next step
            cur_input_ids = torch.cat([cur_input_ids, torch.tensor([[token_id]], device=device)], dim=1)
    return prob

def generate_story(prompt: str, max_length=100, temperature=0.95, top_k=50, top_p=0.9):
    """Generate story autoregressively by sampling."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits
            last_token_logits = logits[0, -1, :] / temperature
            
            # Filter logits with top_k and top_p
            filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered_logits, dim=0)
            
            # Sample next token
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
        
        # If EOS token is generated, stop early
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocab_size)
        top_k >0: keep only top k tokens with highest probability
        top_p >0.0: keep the top tokens with cumulative probability >= top_p
    """
    assert logits.dim() == 1  # logits is 1D tensor
    
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        
        # Remove tokens with cumulative prob above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits

if __name__ == "__main__":
    prompt = "The robot slowly turned its head toward the window, where a red light was"

    next_token_str = "blinking"    
    prob = get_next_token_probability(prompt, next_token_str)
    print(f"word: {next_token_str} probability: {prob}")
    
    #ACTIVIDAD 1: 
    # (1) Calcule el vector de probabilidades de la siguiente palabra dado el contexto de arriba
    # (2) Escriba una funcion que genere las siguientes cinco palabras mas probables de manera autorregresiva.
    # (3) Compare la verosimilitud (promedio) de acuerdo al modelo de las siguientes dos frases
    frase = "Imagination is more important than knowledge"
    frase = "If you think you understand quantum mechanics, you do not understand quantum mechanics."


    print("\nGenerated story:")
    story = generate_story(prompt, max_length=50)
    print(story)

    #ACTIVIDAD 2: 
    # (1) Genere completaciones del prompt de arriba con tres modelos distintos
    model_name = "distilgpt2"  # A lightweight pretrained transformer
    model_name = "EleutherAI/gpt-neo-125M" # A kind of better model
    model_name = "gpt2-medium"
    # (2) Modifique el parametro de temperatura (creciendo y decreciendo) y genere de nuevo las completaciones. 
    # Que sucede cuando la temperatura es muy pequeña y muy grande?
    # (3) Utilice la metodologia de eval-GPT (con ChatGPT libre) para evaluar las historias generadas por los diferentes modelos en las dimensiones 
    # propuestas por Eldan






