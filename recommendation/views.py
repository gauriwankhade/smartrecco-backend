from django.shortcuts import render

# Create your views here.

import requests
from urllib.parse import quote
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated


import chromadb
from chromadb.config import Settings


client = chromadb.PersistentClient(path=".chromadb") 
collection = client.get_or_create_collection("recommendations_openai_v1")  # or any new name


def add_to_vector_db(user_id, query, response, embedding):
    collection.add(
        documents=[query],
        embeddings=[embedding],  # 👈 this ensures consistent 1536-dimension
        metadatas=[{"user_id": user_id, "response": response}],
        ids=[f"{user_id}_{hash(query)}"]
    )




def search_similar_queries(embedding, top_k=3):
    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["distances", "metadatas"]
    )





def get_embedding(text):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer <token>"
    }
    json_data = {
        "model": "text-embedding-ada-002",
        "input": text
    }
    res = requests.post(url, headers=headers, json=json_data)
    return res.json()["data"][0]["embedding"]


class RecommendView(APIView):
  authentication_classes = [TokenAuthentication]
  permission_classes = [IsAuthenticated]
  def post(self, request):
    query = request.data.get("query")
    if not query:
      return Response({"error": "Please type something!"}, status=400)

    try:
      encoded_prompt = quote(query)
      wrapper_user_id = 48
      url = f"https://skillcaptain.app/unicorn/p/llm/openai?userId={wrapper_user_id}&prompt={encoded_prompt}"

      response = requests.get(url, timeout=10)
      result = response.text.strip()

      return Response({"recommendations": result})
    except Exception:
      return Response({"error": "Something went wrong!"}, status=500)



from .utils.vector_db import add_to_vector_db, search_similar_queries
from .utils.embedding import get_embedding

class RecommendWithVectorDbView(APIView):

    def post(self, request):
        query = request.data.get("query")
        if not query:
            return Response({"error": "Please type something"}, status=400)

        embedding = get_embedding(query)

        results = search_similar_queries(embedding, top_k=1)
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Only use vector result if it's a good match
        if metadatas and distances[0] < 0.25:  # 0 = exact match, 1 = completely unrelated
            return Response({
                "recommendation": metadatas[0]["response"],
                "source": "vector-db",
                "similarity": round(1 - distances[0], 2)
            })

        # Fallback to GPT
        encoded_prompt = quote(query)
        wrapper_user_id = 48
        url = f"https://skillcaptain.app/unicorn/p/llm/openai?userId={wrapper_user_id}&prompt={encoded_prompt}"
        response = requests.get(url, timeout=10)
        result = response.text.strip()

        add_to_vector_db(user_id=1, query=query, response=result, embedding=embedding)

        return Response({
            "recommendation": result,
            "source": "gpt"
        })
