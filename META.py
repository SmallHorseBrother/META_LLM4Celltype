# Import necessary packages
import scanpy as sc
import pandas as pd
import numpy as np
import re
import requests
import json
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
import openai
import json
import time


client_openai = openai.OpenAI(
    api_key="your api key"
)


#remove a standard list of uninformative genes
abundant_rnas = [
    "MALAT1",
    "NEAT1",
    "XIST",
    "KCNQ1OT1",
    "RPPH1",
    "RN7SL1",
    "RMRP",
    "SNHG1",
    "MIAT",
    "H19"
]


# Function to remove uninformative genes
def remove_uninformative_genes(adata, abundant_rnas):
    # Remove genes listed in abundant_rnas
    genes_to_keep = ~adata.var_names.isin(abundant_rnas)
    adata = adata[:, genes_to_keep].copy()
    #print(f"Removed {sum(~genes_to_keep)} uninformative genes.")
    return adata

# Function to subset to protein-coding genes (optional)
def subset_to_protein_coding_genes(adata, protein_coding_genes):
    # Subset to protein-coding genes
    genes_to_keep = adata.var_names.isin(protein_coding_genes)
    adata = adata[:, genes_to_keep].copy()
    return adata

# Function to preprocess the data
def preprocess_data(adata):
    # Normalize the data
    sc.pp.normalize_total(adata)

    # Log-transform the data
    sc.pp.log1p(adata)

    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    # # Subset to highly variable genes
    adata = adata[:, adata.var['highly_variable']]  

    # Scale the data
    sc.pp.scale(adata)

    # Perform PCA
    sc.tl.pca(adata, n_comps=50)

    return adata

def prepare_data(adata, abundant_rnas=None, protein_coding_genes=None):
    # 设置 var_names 为基因名称
    adata.var_names = adata.var['feature_name']
    adata.var.index = adata.var_names
    
    # 移除不重要的基因
    if abundant_rnas is not None:
        adata = remove_uninformative_genes(adata, abundant_rnas)
        #print("Removed uninformative genes.")
    # 子集到蛋白编码基因（可选）
    if protein_coding_genes is not None:
        adata = subset_to_protein_coding_genes(adata, protein_coding_genes)
        #print("Subsetted to protein-coding genes.")
    # 数据预处理
    adata = preprocess_data(adata)
    # 计算邻居和 UMAP
    sc.pp.neighbors(adata)

    sc.tl.umap(adata)

    return adata

# Function to find the optimal Leiden resolution
def find_optimal_resolution(adata, min_resolution=0.5 ,max_resolution=0.6, step=0.5):
    best_resolution = None
    best_silhouette_score_value = -1  # Initialize with a negative value
    resolution = min_resolution
    resolutions = []
    silhouette_scores = []

    while resolution <= max_resolution:
        sc.tl.leiden(adata, resolution=resolution)
        num_clusters = adata.obs['leiden'].nunique()
        
        # Compute silhouette score
        try:
            score = silhouette_score(adata.obsm['X_umap'], adata.obs['leiden'])
            print(f"Resolution: {resolution:.2f}, Number of clusters: {num_clusters}, Silhouette Score: {score:.4f}")
            resolutions.append(resolution)
            silhouette_scores.append(score)
            
            if score > best_silhouette_score_value:
                best_silhouette_score_value = score
                best_resolution = resolution
        except ValueError:
            print(f"Resolution: {resolution:.2f} - silhouette_score calculation failed due to insufficient clusters.")
        
        resolution += step
    
    print(f"Best Resolution: {best_resolution} with Silhouette Score: {best_silhouette_score_value:.4f}")
    
    # Apply clustering with the best resolution
    sc.tl.leiden(adata, resolution=best_resolution)

    adata.write('C:\\Users\\MaJianWen\\vscode_file\\chatgpt\\precessed_by_ma\\leiden_liver.h5ad')
    
    return adata, best_resolution

# Function to calculate the majority cell type per cluster and cell-level accuracy
def calculate_majority_celltype_per_cluster(adata, known_cell_type_column='cell_type'):
    # Compute majority cell type per cluster
    majority_celltype_per_cluster = (
        adata.obs.groupby('leiden')[known_cell_type_column]
        .agg(lambda x: x.value_counts().idxmax())
    )
    
    # Map majority cell type to each cell
    adata.obs['majority_celltype'] = adata.obs['leiden'].map(majority_celltype_per_cluster)
    
    # Calculate cell-level accuracy
    correct_cell_labels = adata.obs[known_cell_type_column] == adata.obs['majority_celltype']
    correctly_labeled_cells = correct_cell_labels.sum()
    total_cells = adata.shape[0]
    cell_level_accuracy = correctly_labeled_cells / total_cells
    
    print(f"Cell-level Accuracy (based on majority cell type per cluster): {cell_level_accuracy * 100:.2f}%")
    
    return majority_celltype_per_cluster, cell_level_accuracy


def get_top_marker_genes(adata, top_n=10):
    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test', use_raw=False)
    markers_df = sc.get.rank_genes_groups_df(adata, group=None)
    
    # 直接使用基因名称
    markers_df['gene_name'] = markers_df['names']
    
    # 提取每个簇的前 N 个基因
    markers_df = markers_df.reset_index()
    top_genes = markers_df.groupby('group', as_index=False).apply(lambda x: x.nlargest(top_n, 'logfoldchanges'))
    
    print("Top Marker Genes:")
    print(top_genes[['group', 'gene_name', 'logfoldchanges']])
    
    # 准备 cluster_gene_dict
    cluster_gene_dict = {}
    for group, data in top_genes.groupby('group'):
        gene_list = data['gene_name'].tolist()
        cluster_gene_dict[str(group)] = gene_list

    
    return cluster_gene_dict


import gseapy as gp
import time
import random





# Function to create prompts with full context
def create_prompt_more_context(cluster_gene_dict, target_cluster, previous_annotations, known_cell_types=None, prompt_template=None,tissue='liver'):
    """
    Generate a prompt for LLM with all cluster genes as context and previous annotations.
    """
    all_clusters_genes = "\n".join(
        [f"Cluster {cluster}: {', '.join(genes)}" for cluster, genes in cluster_gene_dict.items()]
    )

    previous_results = "\n".join(
        [f"Cluster {cluster}: {annotation} (Confidence: {confidence})"
         for cluster, (annotation, confidence) in previous_annotations.items()]
    ) if previous_annotations else "No previous annotations available."

    if prompt_template is None:
        prompt = (
            f"You are a cell biology expert specializing in identifying cell types based on marker genes. "
            f"The following clusters and their associated marker genes are given for reference:\n\n"
            f"{all_clusters_genes}\n\n"
            f"Previously annotated clusters (if any):\n{previous_results}\n\n"
            f"Now, focus on the target cluster:\n"
            f"Cluster {target_cluster}:\n"
            f"Marker genes (extracted from differential expression analysis and ranked by descending order of differential expression):\n"
            f"{', '.join(cluster_gene_dict[target_cluster])}\n\n"
            f"These cells are from {tissue} tissue.\n\n"
            "Please predict the cell type for the target cluster based on these marker genes. "
            "Ensure the predicted cell type is a valid name following ontological conventions. "
            "Also provide a confidence score (from 0 to 1) for your prediction.\n\n"
            "Final Output:\n"
            "1. Predicted Cell Type: [cell type name]\n"
            "2. Confidence: [confidence score]"
        )
    else:
        prompt = prompt_template.format(
            all_clusters_genes=all_clusters_genes,
            target_cluster=target_cluster,
            target_genes=', '.join(cluster_gene_dict[target_cluster]),
            previous_results=previous_results
        )
    return prompt



# Function to create prompts with full context
def create_prompt_more_context_cot(cluster_gene_dict, target_cluster, previous_annotations, known_cell_types=None, prompt_template=None,tissue='liver'):
    """
    Generate a prompt for LLM with all cluster genes as context and previous annotations.
    """
    all_clusters_genes = "\n".join(
        [f"Cluster {cluster}: {', '.join(genes)}" for cluster, genes in cluster_gene_dict.items()]
    )

    previous_results = "\n".join(
        [f"Cluster {cluster}: {annotation} (Confidence: {confidence})"
         for cluster, (annotation, confidence) in previous_annotations.items()]
    ) if previous_annotations else "No previous annotations available."

    if prompt_template is None:
        prompt = (
            f"You are a cell biology expert specializing in identifying cell types based on marker genes. "
            f"The following clusters and their associated marker genes are given for reference:\n\n"
            f"{all_clusters_genes}\n\n"
            f"Previously annotated clusters (if any):\n{previous_results}\n\n"
            f"Now, focus on the target cluster:\n"
            f"Cluster {target_cluster}:\n"
            f"Marker genes (extracted from differential expression analysis and ranked by descending order of differential expression):\n"
            f"{', '.join(cluster_gene_dict[target_cluster])}\n\n"
            f"These cells are from {tissue} tissue.\n\n"
            "Please predict the cell type for the target cluster based on these marker genes.\n"
            "Think step by step,give the reasoning process; .\n\n"
            "please give final conclusion like this:\n"
            "1. Predicted Cell Type: [cell type name]\n"
            "2. Confidence: [confidence score]"
        )
    else:
        prompt = prompt_template.format(
            all_clusters_genes=all_clusters_genes,
            target_cluster=target_cluster,
            target_genes=', '.join(cluster_gene_dict[target_cluster]),
            previous_results=previous_results
        )
    return prompt




# Function to extract cell type from LLM response
def extract_cell_type(response, known_cell_types):
    # Improved regex to capture the cell type after 'Final Answer:'
    match = re.search(r'Final Answer:\s*([A-Za-z0-9\s\-\(\)]+)', response, re.IGNORECASE | re.MULTILINE)
    if match:
        cell_type = match.group(1).strip()
        # Validate against known cell types
        if any(cell_type.lower() == ct.lower() for ct in known_cell_types):
            return cell_type
    return None

 #获取think内容并从</think>后面提取
def extract_answer_from_think(cleaned_response):
    # 查找</think>的结束位置
    think_end_index = cleaned_response.find('</think>')
    
    if think_end_index != -1:
        # 从</think>后面开始提取
        answer = cleaned_response[think_end_index+8:].strip()
    else:
        # 如果没有找到</think>标签，可以直接返回整个响应
        answer = cleaned_response.strip()
    
    return answer


def intelligent_extraction(response, known_cell_types=None, model_name="llama3.1:8b-instruct-q8_0", url="http://localhost:11434/api/generate"):
    # Clean the response to keep only the part after 'Final Answer:'
    final_answer_text = re.search(r'Final Answer:\s*(.*)', response, re.IGNORECASE | re.MULTILINE)
    if final_answer_text:
        cleaned_response = final_answer_text.group(1).strip()
    else:
        cleaned_response = response.strip()  # Use the original response as fallback

    # 从think内容之后提取答案
    cleaned_response = extract_answer_from_think(cleaned_response)

    # Construct prompt for intelligent extraction
    extract_prompt = (
        f"Extract the cell type from the following response. The cell type should be a valid biological cell type "
        f"following standard ontological naming conventions. Please output only the exact cell type name.\n\n"
        f"Response: {cleaned_response}\n\n"
        "Extracted cell type:"
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the correct cell type based on ontological standards.."},
            {"role": "user", "content": extract_prompt}
        ],
        "temperature": 0.1,
        "timeout": 60
    }
    
    response = get_response_openai(data)
    return response.strip()


def intelligent_extraction_confidence(response, known_cell_types=None, model_name="llama3.1:8b-instruct-q8_0", url="http://localhost:11434/api/generate"):
    """
    Extracts the cell type and confidence from the LLM response.
    """
    # Extract final answer text
    final_answer_match = re.search(r'Final Answer:\s*(.*)', response, re.IGNORECASE | re.MULTILINE)
    if final_answer_match:
        cleaned_response = final_answer_match.group(1).strip()
    else:
        cleaned_response = response.strip()  # Fallback to the entire response

    # 从think内容之后提取答案
    cleaned_response = extract_answer_from_think(cleaned_response)

    # Construct prompt for intelligent extraction
    extract_prompt = (
        f"Extract the cell type and confidence from the following response. The cell type should follow standard ontological naming conventions. "
        f"If no confidence is provided, estimate one based on the clarity of the response (scale 0-1).\n\n"
        f"Response: {cleaned_response}\n\n"
        "Extracted Cell Type: [name]\nConfidence: [value]"
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract cell type and confidence accurately."},
            {"role": "user", "content": extract_prompt}
        ],
        "temperature": 0.1,
        "timeout": 60
    }

    response = get_response_openai(data)  # Replace this with actual LLM API call
    # Parsing response
    cell_type_match = re.search(r'Extracted Cell Type:\s*(.*)', response, re.IGNORECASE | re.MULTILINE)
    confidence_match = re.search(r'Confidence:\s*(\d+(\.\d+)?)', response, re.IGNORECASE | re.MULTILINE)

    cell_type = cell_type_match.group(1).strip() if cell_type_match else None
    confidence = float(confidence_match.group(1)) if confidence_match else 0.5  # Default confidence to 0.5 if not found

    return cell_type, confidence


url_generate = "http://localhost:11434/api/generate" # Define Ollama API URL

import json
import subprocess

def get_response_ollama(data, model_name="llama2"):
    try:
        # Extract prompt from data
        prompt = data.get("prompt", "")

        # Prepare the ollama command
        command = [
            "ollama",
            "run",
            model_name,  # Model name as the first argument
            prompt  # Pass prompt as the prompt argument
        ]

        # Run ollama command
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=True
        )

        full_response = result.stdout.strip()

        # Assume that the output starts after the prompt echo
        # Remove the echoed prompt and extract the response
        # This part depends on how Ollama outputs the response
        # You might need to adjust this logic based on Ollama's output format
        full_response = full_response.split("\n", 1)[-1].strip()

        return full_response

    except subprocess.CalledProcessError as e:
        print(f"Error executing ollama command: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Function to get response from OpenAI API
def get_response_openai(data):
    completion = client_openai.chat.completions.create(**data)
    response_content = completion.choices[0].message.content
    return response_content.strip()

import re

def parse_corrected_annotations(response):
    """
    Parse the corrected annotations from the LLM response.
    """
    corrected_annotations = {}
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith('cluster'):
            # Match pattern 'Cluster X: Cell Type Name'
            match = re.match(r'Cluster\s+(\d+):\s*(.+)', line, re.IGNORECASE)
            if match:
                cluster_id_str, cell_type = match.groups()
                cluster_id = int(cluster_id_str)
                corrected_annotations[cluster_id] = cell_type.strip()
            else:
                print(f"Warning: Line format is incorrect: '{line}'")
        else:
            print(f"Warning: Line does not start with 'Cluster': '{line}'")
    return corrected_annotations



def annotate_clusters_with_llm_more_context(adata, cluster_gene_dict, known_cell_types, model_name="gpt-4o-mini", prompt_template=None,tissue='liver',temperature = 0.1,top_p = 1):
    """
    Annotate clusters using LLM with full context and progressive annotations.
    """
    cluster_annotations = {}

    # for cluster, genes in cluster_gene_dict.items():
    #     print(f"Cluster {cluster} - Genes: {genes}")


    for cluster, genes in cluster_gene_dict.items():
        #print(f"Processing Cluster {cluster}...")

        # Generate the prompt with all clusters and previous annotations as context
        prompt = create_prompt_more_context(cluster_gene_dict, cluster, cluster_annotations, known_cell_types, prompt_template,tissue)

        # Call the LLM API
        data = {
            "model": model_name, # Use model_name parameter
            "prompt": prompt, # Use 'prompt' key for Ollama
            "system": "You are an expert in cell type annotation." # Add system prompt
        }

        response = get_response_ollama(data, model_name=model_name)#, temperature=temperature, top_p=top_p)  # Use get_response_ollama and pass model_name, temperature, top_p

        #print(f"LLM Response for Cluster {cluster}:\n", response.strip())

        # Extract the cell type from the response
        predicted_cell_type, confidence = intelligent_extraction_confidence(response, known_cell_types, model_name=model_name)

        cluster_annotations[cluster] = (predicted_cell_type, confidence)
        print(f"Predicted Cell Type for Cluster {cluster}: {predicted_cell_type} (Confidence: {confidence})")

    print("Cluster Annotations before correct:", cluster_annotations)


    return cluster_annotations


# Function to annotate clusters with context
def annotate_clusters_with_llm_more_context_cot(adata, cluster_gene_dict, known_cell_types, model_name="gpt-4o-mini", prompt_template=None,tissue='liver',temperature=0.1, top_p=1.0):
    """
    Annotate clusters using LLM with full context and progressive annotations.
    """
    cluster_annotations = {}

    # for cluster, genes in cluster_gene_dict.items():
    #     print(f"Cluster {cluster} - Genes: {genes}")


    for cluster, genes in cluster_gene_dict.items():
        #print(f"Processing Cluster {cluster}...")

        # Generate the prompt with all clusters and previous annotations as context
        prompt = create_prompt_more_context_cot(cluster_gene_dict, cluster, cluster_annotations, known_cell_types, prompt_template,tissue)

        # Call the LLM API
        data = {
            "model": model_name, # Use model_name parameter
            "prompt": prompt, # Use 'prompt' key for Ollama
            "system": "You are an expert in cell type annotation." # Add system prompt
        }

        response = get_response_ollama(data, model_name=model_name)#, temperature=temperature, top_p=top_p)  # Use get_response_ollama and pass model_name, temperature, top_p

        #print(f"LLM Response for Cluster {cluster}:\n", response.strip())

        # Extract the cell type from the response
        predicted_cell_type, confidence = intelligent_extraction_confidence(response, known_cell_types, model_name=model_name)

        cluster_annotations[cluster] = (predicted_cell_type, confidence)
        print(f"Predicted Cell Type for Cluster {cluster}: {predicted_cell_type} (Confidence: {confidence})")

    print("Cluster Annotations before correct:", cluster_annotations)

    return cluster_annotations



import re

# Helper function to normalize strings
def normalize_string(s):
    return re.sub(r'\W+', '', s).lower().strip()


def extract_format_from_response(strong_model_response):
    """
    This function takes the raw response from the strong LLM (GPT-4o) and extracts the required fields.
    It formats the response to match the expected structure: 'binary', 'categorical', 'categorical_improved'.
    """
    prompt = (
        "The previous response from a weaker model is incomplete and missing some key values. "
        "Please extract the values for the following keys and format them correctly:\n\n"
        "1. 'Binary Mode' (yes/no)\n"
        "2. 'Categorical Mode' (perfect match/partial match/no match)\n"
        "3. 'Categorical Improved Mode' (perfect match/partial match/no match)\n\n"
        "Here is the incomplete response:\n"
        f"{strong_model_response}\n\n"
        "Please format the extracted values into this structure:\n"
        "Binary Mode: yes/no\n"
        "Categorical Mode: perfect match/partial match/no match\n"
        "Categorical Improved Mode: perfect match/partial match/no match"
    )

    # Prepare the data for LLM call to extract the missing information
    data = {
        "model": "gpt-4o",  # Use GPT-4o as the stronger model
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in cell type annotation. Please extract missing information from an incomplete response."
            },
            {"role": "user", "content": prompt}
        ],
        "timeout": 60
    }

    # Call the strong LLM to extract the required keys
    response = get_response_openai(data)
    response = response.strip()

    # Process the response to extract the necessary keys
    extracted_results = {}
    for line in response.splitlines():
        if "Binary Mode:" in line:
            extracted_results["binary"] = line.split(":", 1)[1].strip()
        elif "Categorical Mode:" in line:
            extracted_results["categorical"] = line.split(":", 1)[1].strip()
        elif "Categorical Improved Mode:" in line:
            extracted_results["categorical_improved"] = line.split(":", 1)[1].strip()

    return extracted_results


def unified_ai_compare_with_modes(ground_truth, predicted):
    """
    Compare cell type labels under three modes (binary, categorical, categorical_improved) and return all results.

    Args:
        ground_truth (str): Ground truth label.
        predicted (str): Predicted cell type label.

    Returns:
        dict: A dictionary containing the results for each mode:
            - "binary": Yes/No result for coarse-grained match.
            - "categorical": Strict classification (perfect match/partial match/no match).
            - "categorical_improved": Flexible classification allowing perfect match for finer predictions.
    """
    # Prepare the unified prompt
    prompt = (
        f"Compare the predicted cell type label with the ground truth label:\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {predicted}\n\n"

        "Evaluate the match under three different modes:\n\n"

        "1. **Binary Mode (Coarse Match)**:\n"
        "- Return 'yes' if the labels belong to the same broad category.\n"
        "- Return 'no' if they are entirely different.\n"
        "Examples:\n"
        "   1) B cell, Plasma cell -> yes\n"
        "   2) Macrophage, Endothelial cell -> no\n\n"

        "2. **Categorical Mode (Strict Match)**:\n"
        "- Classify the match into:\n"
        "   * 'perfect match': Labels are semantically identical.\n"
        "   * 'partial match': Labels belong to the same broad category but differ in specificity.\n"
        "   * 'no match': Labels have no meaningful overlap.\n"
        "Examples:\n"
        "   1) B cell, Plasma cell -> partial match\n"
        "   2) Macrophage, Endothelial cell -> no match\n"
        "   3) Macrophage, Macrophage -> perfect match\n\n"

        "3. **Categorical Improved Mode (Flexible Match)**:\n"
        "- Classify the match into:\n"
        "   * 'perfect match': Same broad category, and Predicted is either identical to or more specific than Ground Truth.\n"
        "   * 'partial match': Same broad category, but Predicted is less specific or in a different subset.\n"
        "   * 'no match': Labels have no meaningful overlap.\n"
        "Examples:\n"
        "   1) B cell, Plasma cell -> perfect match\n"
        "   2) Macrophage, Kupffer cell -> perfect match\n"
        "   3) Macrophage, Endothelial cell -> no match\n\n"

        "Now provide results for all three modes in this format:\n"
        "Binary Mode: yes/no\n"
        "Categorical Mode: perfect match/partial match/no match\n"
        "Categorical Improved Mode: perfect match/partial match/no match\n"
    )

    # Prepare the data for LLM call
    data = {
        "model": "gpt-4o-mini",  # Use your specific model
        "messages": [
            {
                "role": "system",
                "content": "You are a molecular biologist tasked with comparing cell type labels under different matching rules."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "timeout": 60
    }

    # Call LLM and extract response
    response = get_response_openai(data)

    print("LLM Response:", response)  # 调试输出，查看 LLM 返回的原始内容

    response = response.strip()

    # Process response into dictionary format
    results = {}
    for line in response.splitlines():
        if "Binary Mode:" in line:
            results["binary"] = line.split(":", 1)[1].strip()
        elif "Categorical Mode:" in line:
            results["categorical"] = line.split(":", 1)[1].strip()
        elif "Categorical Improved Mode:" in line:
            results["categorical_improved"] = line.split(":", 1)[1].strip()

    
    # If any key is missing or the results is empty, return default values as "0"
    if not results or 'binary' not in results or 'categorical' not in results or 'categorical_improved' not in results:
        print("Warning: LLM response parsing failed or missing keys. Returning default values.")
        return {
            "binary": "0",
            "categorical": "0",
            "categorical_improved": "0"
        }

    
    return results

def calculate_llm_annotation_accuracy(adata, cluster_annotations, majority_celltype_per_cluster, known_cell_type_column='cell_type'):
    """
    Calculates the annotation accuracy of LLM-predicted cell types compared to ground truth labels,
    supporting three evaluation metrics: binary, categorical, and categorical improved.
    Also computes the proportion of perfect match, partial match, and no match for categorical metrics.
    
    Args:
        adata (AnnData): The annotated data object.
        cluster_annotations (dict): Cluster to predicted cell type mappings.
        majority_celltype_per_cluster (Series): Majority cell type per cluster based on ground truth.
        known_cell_type_column (str): Column name for ground truth cell types in adata.obs.
    
    Returns:
        dict: A dictionary containing the results for each evaluation metric:
            - Binary: Overall accuracy.
            - Categorical: Overall accuracy and match proportions.
            - Categorical Improved: Overall accuracy and match proportions.
    """

    # Normalize cluster_annotations to map cluster to cell type string
    normalized_cluster_annotations = {}
    for cluster, annotation in cluster_annotations.items():
        if isinstance(annotation, tuple):
            predicted_cell_type = annotation[0]
        else:
            predicted_cell_type = annotation
        normalized_cluster_annotations[str(cluster)] = predicted_cell_type
    
    cluster_annotations = normalized_cluster_annotations

    # Add the LLM annotations to the AnnData object
    adata.obs['predicted_celltype'] = adata.obs['leiden'].map(cluster_annotations)
    
    # Initialize variables to store total scores and match counts
    binary_scores = []
    categorical_scores = []
    categorical_improved_scores = []

    # Match counts for categorical metrics
    categorical_counts = {'perfect': 0, 'partial': 0, 'no': 0}
    categorical_improved_counts = {'perfect': 0, 'partial': 0, 'no': 0}

     # Initialize per_cluster_match_results_dict to store match results per cluster
    per_cluster_match_results_dict = {}
    
    # Iterate over each cluster
    for cluster in adata.obs['leiden'].unique():
        # Get the ground truth majority cell type for the cluster
        ground_truth_majority = majority_celltype_per_cluster.get(cluster, None)
        # Get the predicted cell type for the cluster
        predicted_cell_type = cluster_annotations.get(cluster, None)
        
        if not ground_truth_majority or not predicted_cell_type:
            continue  # Skip if missing data
        
        # Use AI to compare the cell types under unified function
        match_results = unified_ai_compare_with_modes(ground_truth_majority, predicted_cell_type)

        print(f"groud_truth_majority: {ground_truth_majority}, predicted_cell_type: {predicted_cell_type}, match_results: {match_results}")

        # **Store match_results per cluster**
        per_cluster_match_results_dict[cluster] = {
            "binary_match": match_results['binary'],
            "categorical_match": match_results['categorical'],
            "categorical_improved_match": match_results['categorical_improved']
        }

        # Binary mode score
        binary_scores.append(1.0 if match_results['binary'] == 'yes' else 0.0)

        # Categorical mode score
        categorical_match = match_results['categorical']
        if categorical_match == 'perfect match':
            categorical_scores.append(1.0)
            categorical_counts['perfect'] += 1
        elif categorical_match == 'partial match':
            categorical_scores.append(0.5)
            categorical_counts['partial'] += 1
        else:
            categorical_scores.append(0.0)
            categorical_counts['no'] += 1

        # Categorical improved mode score
        categorical_improved_match = match_results['categorical_improved']
        if categorical_improved_match == 'perfect match':
            categorical_improved_scores.append(1.0)
            categorical_improved_counts['perfect'] += 1
        elif categorical_improved_match == 'partial match':
            categorical_improved_scores.append(0.5)
            categorical_improved_counts['partial'] += 1
        else:
            categorical_improved_scores.append(0.0)
            categorical_improved_counts['no'] += 1

    # Calculate overall scores and proportions
    def calculate_average_and_proportions(scores, counts):
        total = len(scores)
        if total == 0:
            return 0.0, {'perfect': 0.0, 'partial': 0.0, 'no': 0.0}
        
        accuracy = sum(scores) / total
        proportions = {key: (value / total) * 100 for key, value in counts.items()}
        return accuracy, proportions

    # Binary mode results
    binary_accuracy = sum(binary_scores) / len(binary_scores) if binary_scores else 0.0

    # Categorical mode results
    categorical_accuracy, categorical_proportions = calculate_average_and_proportions(categorical_scores, categorical_counts)

    # Categorical improved mode results
    categorical_improved_accuracy, categorical_improved_proportions = calculate_average_and_proportions(categorical_improved_scores, categorical_improved_counts)

    # Return results
    return {
        'binary': {
            'accuracy': binary_accuracy * 100
        },
        'categorical': {
            'accuracy': categorical_accuracy * 100,
            'proportions': categorical_proportions
        },
        'categorical_improved': {
            'accuracy': categorical_improved_accuracy * 100,
            'proportions': categorical_improved_proportions
        }
    }, per_cluster_match_results_dict


import pandas as pd
import pickle

def save_accuracy_table(accuracy_data, output_file="cluster_accuracy_liver.csv"):
    """
    Save the cluster annotation accuracy for different versions and tissues to a CSV file.
    """
    df = pd.DataFrame(accuracy_data)
    df.to_csv(output_file, index=False)

def load_preprocessed_data(tissue):
    """
    Load preprocessed data from local files.
    """
    print(f"Loading preprocessed data for {tissue}...")
    adata = sc.read(f'adata_{tissue}_processed.h5ad')
    with open(f'{tissue}_metadata.pkl', 'rb') as f:
        tissue_data = pickle.load(f)
    print(f"Finished loading data for {tissue}.")
    return adata, tissue_data


def analyze_tissue(tissue, abundant_rnas, known_cell_type_column, model_name, prompt_template, run_num):
    """
    Perform analysis for a specific tissue using preprocessed data and save results incrementally.
    Includes run number in filenames.
    """
    # Load preprocessed data
    adata, tissue_data = load_preprocessed_data(tissue)

    # Extract necessary information from preprocessed data
    majority_celltype_per_cluster = tissue_data["majority_celltype_per_cluster"]
    cluster_gene_dict = tissue_data["cluster_gene_dict"]
    known_cell_types = tissue_data["known_cell_types"]
    cluster_count = len(adata.obs['leiden'].unique()) # Initialize cluster count
    # Record accuracy for each version
    accuracy_data = []
    annotation_data = []

    annotation_versions = [

        ("more_context",annotate_clusters_with_llm_more_context),
        ("more_context_cot", annotate_clusters_with_llm_more_context_cot),

    ]

    for version, annotate_func in annotation_versions:
        annotation_result = annotate_func(
            adata, cluster_gene_dict, known_cell_types, model_name=model_name,
            prompt_template=prompt_template, tissue=tissue
        )

        # Calculate and record accuracy for original annotations
        accuracy_results, per_cluster_match_results  = calculate_llm_annotation_accuracy(
            adata, annotation_result, majority_celltype_per_cluster, known_cell_type_column
        )

        print(f"Accuracy Results for {version} on {tissue} (Run {run_num}):")
        print(accuracy_results)

        # Save accuracy data for original annotations
        accuracy_data_entry = {
            "Version": version,
            "Tissue": tissue,
            "Run": run_num,
            "Cluster Count": cluster_count,
            "Binary Accuracy": accuracy_results['binary']['accuracy'],
            "Categorical Accuracy": accuracy_results['categorical']['accuracy'],
            "Categorical Improved Accuracy": accuracy_results['categorical_improved']['accuracy'],
            "Categorical Perfect Match": accuracy_results['categorical']['proportions']['perfect'],
            "Categorical Partial Match": accuracy_results['categorical']['proportions']['partial'],
            "Categorical No Match": accuracy_results['categorical']['proportions']['no'],
            "Categorical Improved Perfect Match": accuracy_results['categorical_improved']['proportions']['perfect'],
            "Categorical Improved Partial Match": accuracy_results['categorical_improved']['proportions']['partial'],
            "Categorical Improved No Match": accuracy_results['categorical_improved']['proportions']['no'],
        }
        accuracy_data.append(accuracy_data_entry)

        # Record annotation results for original annotations
        for cluster in annotation_result:
            match_result = per_cluster_match_results.get(cluster, {})
            annotation_data.append({
                "Version": version,
                "Tissue": tissue,
                "Run": run_num,
                "Cluster": cluster,
                "Predicted Cell Type": annotation_result[cluster],
                "Ground Truth": majority_celltype_per_cluster.get(cluster),
                "Binary Match": match_result.get("binary_match"),
                "Categorical Match": match_result.get("categorical_match"),
                "Categorical Improved Match": match_result.get("categorical_improved_match"),
            })

        tissue_name = tissue  # Use single tissue name for file naming
        file_name_acc = f"cluster_accuracy_{tissue_name}_{model_name}_run{run_num}_{version}.csv"  # Generate the file name with version
        file_name_annot = f"cluster_annotations_{tissue_name}_{model_name}_run{run_num}_{version}.csv"  # Generate the file name with version

        # Save accuracy and annotation data for each version and tissue immediately
        save_accuracy_table([accuracy_data_entry], file_name_acc) # Save single entry accuracy data
        annotation_df = pd.DataFrame(annotation_data) # Save all annotation data so far (for current tissue and version)
        annotation_df.to_csv(file_name_annot, index=False)
        annotation_data = [] # Clear annotation data for next version in the same tissue


    return accuracy_data, annotation_data # Return accuracy and annotation data for potential further processing if needed


def run_analysis_pipeline(tissue_list, abundant_rnas, known_cell_type_column, model_name, prompt_template, num_runs=1):
    for run_num in range(2, num_runs + 1): # Loop for the desired number of runs
        for tissue in tissue_list:
            print(f"Starting analysis for tissue: {tissue}, Run: {run_num}")
            analyze_tissue(
                tissue, abundant_rnas, known_cell_type_column, model_name, prompt_template, run_num=run_num
            )
            print(f"Finished analysis and saved results for tissue: {tissue}, Run: {run_num}") # Confirmation message after each tissue is processed


run_analysis_pipeline(
    tissue_list=['liver','pancreas','uterus'],
    #tissue_list=['liver'],
    abundant_rnas=abundant_rnas,
    known_cell_type_column='cell_type',
    model_name="deepseek-r1:1.5b",
    prompt_template=None,
    num_runs=4 # Run 3 times
)
