struct MPI_Comm
{
    int dummy;
};

struct MPI_Datatype
{
    int dummy;
};

struct MPI_Status
{
    int dummy;
};

struct MPI_Op {
    int dummy;
};

struct MPI_Group {
    int dummy;
};

struct MPI_Op MPI_MAX;
struct MPI_Op MPI_MIN;
struct MPI_Op MPI_SUM;
struct MPI_Op MPI_PROD;
struct MPI_Op MPI_LAND;
struct MPI_Op MPI_LOR;
struct MPI_Op MPI_BAND;
struct MPI_Op MPI_BOR;
struct MPI_Op MPI_MAXLOC;
struct MPI_Op MPI_MINLOC;

struct MPI_Comm MPI_COMM_WORLD;

struct MPI_Status MPI_STATUS_IGNORE;

struct MPI_Datatype MPI_INT;
struct MPI_Datatype MPI_FLOAT;
struct MPI_Datatype MPI_DOUBLE;

int MPI_ANY_SOURCE;

void MPI_Init(int* argc, char*** argv);

void MPI_Finalize();

void MPI_Comm_size(struct MPI_Comm communicator, int* size);

void MPI_Comm_Rank(struct MPI_Comm communicator, int* rank);

void MPI_Get_processor_name(char* name, int* name_length);

void MPI_Finalize();

void MPI_Send(void* data, int count, struct MPI_Datatype datatype, int destination, int tag, struct MPI_Comm communicator);

void MPI_Recv(void* data, int count, struct MPI_Datatype datatype, int source, int tag, struct MPI_Comm communicator, struct MPI_Status* status);

void MPI_Get_count(struct MPI_Status* status, struct MPI_Datatype datatype, int* count);

void MPI_Probe(int source, int tag, struct MPI_Comm comm, struct MPI_Status* status);

void MPI_Bcast(void* data, int count,struct MPI_Datatype datatype, int root, struct MPI_Comm communicator);

void MPI_Barrier(struct MPI_Comm communicator);

void MPI_Scatter(void* send_data, int send_count,  struct MPI_Datatype send_datatype, void* recv_data, int recv_count,  struct MPI_Datatype recv_datatype, int root,  struct MPI_Comm communicator);

void MPI_Gather(void* send_data, int send_count,  struct MPI_Datatype send_datatype, void* recv_data, int recv_count,  struct MPI_Datatype recv_datatype, int root,  struct MPI_Comm communicator);

void MPI_Allgather( void* send_data, int send_count,  struct MPI_Datatype send_datatype, void* recv_data, int recv_count,  struct MPI_Datatype recv_datatype,  struct MPI_Comm communicator);

void MPI_Reduce( void* send_data, void* recv_data, int count,  struct MPI_Datatype datatype,  struct MPI_Op op, int root,  struct MPI_Comm communicator);

void MPI_Allreduce( void* send_data, void* recv_data, int count,  struct MPI_Datatype datatype,  struct MPI_Op op,  struct MPI_Comm communicator);

void MPI_Comm_split( struct MPI_Comm comm, int color, int key, struct MPI_Comm* newcomm);

void MPI_Comm_create( struct MPI_Comm comm, struct MPI_Group group,  struct MPI_Comm* newcomm);

void MPI_Comm_group( struct MPI_Comm comm, struct MPI_Group* group);

void MPI_Group_union(struct MPI_Group group1, struct MPI_Group group2, struct MPI_Group* newgroup);

void MPI_Group_intersection(struct MPI_Group group1, struct MPI_Group group2, struct MPI_Group* newgroup);

void MPI_Comm_create_group(struct MPI_Comm comm, struct MPI_Group group,int tag,  struct MPI_Comm* newcomm);

void MPI_Group_incl(struct MPI_Group group, int n, const int ranks[], struct MPI_Group* newgroup);

