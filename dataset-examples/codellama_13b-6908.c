//Code Llama-13B DATASET v1.0 Category: Graph representation ; Style: curious
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  int value;
  struct Node *next;
} Node;

typedef struct Graph {
  Node *head;
} Graph;

void addNode(Graph *graph, int value) {
  Node *newNode = malloc(sizeof(Node));
  newNode->value = value;
  newNode->next = graph->head;
  graph->head = newNode;
}

void removeNode(Graph *graph, int value) {
  Node *curr = graph->head;
  Node *prev = NULL;
  while (curr != NULL && curr->value != value) {
    prev = curr;
    curr = curr->next;
  }
  if (curr == NULL) {
    return;
  }
  if (prev == NULL) {
    graph->head = curr->next;
  } else {
    prev->next = curr->next;
  }
  free(curr);
}

void printGraph(Graph *graph) {
  Node *curr = graph->head;
  while (curr != NULL) {
    printf("%d ", curr->value);
    curr = curr->next;
  }
  printf("\n");
}

int main() {
  Graph graph;
  graph.head = NULL;
  addNode(&graph, 1);
  addNode(&graph, 2);
  addNode(&graph, 3);
  addNode(&graph, 4);
  printGraph(&graph);
  removeNode(&graph, 2);
  printGraph(&graph);
  return 0;
}