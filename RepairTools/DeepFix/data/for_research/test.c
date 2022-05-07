#include <stdio.h>
#include <stdlib.h>
/*int length=0;
// to store the link list
struct node
{
    int data;
    struct node* next;
};

// insertion in the list "head", at the END
struct node* insert(struct node* head, int data)
{
    struct node* n=(struct node*)malloc(sizeof(struct node));
    n->next=NULL;
    n->data=data;
    if(head==NULL)
    {
        length++;
        return n;
    }
    struct node* tmp=head;
    while(tmp->next!=NULL)
        {
            length++;
            tmp=tmp->next;
        }
    tmp->next=n;
    return head;
}

//delete
struct node *delete(struct node *pnode, struct node*ppnode)
{
struct node *t;
if (ppnode)
ppnode->next = pnode->next;
t = ppnode? ppnode: pnode->next;
free (pnode);
return t;
}

struct node *pointer(struct node*head,int index)
{
    struct node *curr;
    curr=head;
    int count=0;
    while(curr!=NULL)
    {
        if(count==index-1)
        return curr;
        count++;
    }
}

// print the list "head"
void print(struct node* head)
{
    struct node *curr;
    curr=head;
    while(curr!=NULL)  {
        printf("%d ",curr->data);
        curr=curr->next;
    }
    printf("X\n");
    return;
}


int main() {
	// Fill this area with your code.
	struct node*head;
	struct node list;
	
int data;
	scanf("%d",&data);
	while(data!=-1)
	{
	    head.next=insert(head.next,data);
	    scanf("%d",&data);
	}
	int n;
	scanf("%d",&n);
	struct node*ppnode=pointer(head,length+1-n);
	struct node*pnode=ppnode.next;
	//delete(pointer(head->next,length+1-n)->next,pointer(head,length+1-n));
	
	return 0;
}*/



struct node{
    int data;
    struct node struct * next ;
} ;

int main(){
    struct node*head;
    struct node list;
    *head=list;
}

