from django.shortcuts import render

# Create your views here.

import requests
from urllib.parse import quote
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

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
