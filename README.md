# generacion_de_texto_llm
- Se crea un chatbot para generación de texto alimentado por documentos de word .docx
- El archivo app.pý contiene la aplicación que corre en fastapi, contiene el prompt.
- se ejecuta con la instrucción fastapi dev app.py
- se escogió un tamaño de chunk igual 1000 para dar más contexto al modelo
- se escogió una tamaño de solapamiento en los chunks igual a 300 para tener secciones independientes  