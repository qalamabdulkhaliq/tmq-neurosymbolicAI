"""
F7 SPARQL Endpoint.
Read-only Flask server over quran_root_ontology_v3.ttl.
POST /sparql with form param 'query', returns JSON results.
Ontology is never modified.
"""
import threading
import logging
from flask import Flask, request, jsonify
import rdflib

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


class SPARQLServer:
    def __init__(self, ttl_path: str, port: int = 5820):
        self._ttl_path = ttl_path
        self._port = port
        self._graph = None
        self._app = None
        self._thread = None
        self.running = False

    def _load_graph(self):
        self._graph = rdflib.ConjunctiveGraph()
        self._graph.parse(self._ttl_path, format="turtle")

    def _build_app(self) -> Flask:
        app = Flask("sparql_endpoint")

        @app.post("/sparql")
        def sparql_query():
            query = request.form.get("query", "").strip()
            if not query:
                return jsonify({"error": "no query"}), 400
            try:
                results = self._graph.query(query)
                bindings = []
                for row in results:
                    binding = {}
                    for var in results.vars:
                        val = getattr(row, str(var), None)
                        if val is not None:
                            binding[str(var)] = {"value": str(val)}
                    bindings.append(binding)
                return jsonify({"results": {"bindings": bindings}})
            except Exception as e:
                return jsonify({"error": str(e)}), 400

        return app

    def start(self):
        self._load_graph()
        self._app = self._build_app()
        self._thread = threading.Thread(
            target=lambda: self._app.run(
                port=self._port, use_reloader=False, threaded=True
            ),
            daemon=True,
        )
        self._thread.start()
        self.running = True

    def stop(self):
        self.running = False
