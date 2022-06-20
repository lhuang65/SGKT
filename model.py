# encoding:utf-8
import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator
import math
import tensorflow.contrib as tf_contrib

class SGKT(object):
    def __init__(self, args):
        self.args = args
        self.hidden_neurons = args.hidden_neurons
        self.max_step = args.max_step - 1
        self.feature_answer_size = args.feature_answer_size
        self.field_size = args.field_size
        self.embedding_size = args.embedding_size
        self.dropout_keep_probs = eval(args.dropout_keep_probs)
        self.select_index = args.select_index
        self.hist_neighbor_num = args.hist_neighbor_num  # M
        self.next_neighbor_num = args.next_neighbor_num  # N
        self.lr = args.lr
        self.n_hop = args.n_hop
        self.question_neighbor_num = args.question_neighbor_num
        self.skill_neighbor_num = args.skill_neighbor_num
        self.question_neighbors = args.question_neighbors
        self.skill_neighbors = args.skill_neighbors
        for i in range(len(self.skill_neighbors)):
            for j in range(len(self.question_neighbors[i])):
                self.question_neighbors[i][j]=self.skill_neighbors[i][j]
        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep prob
        self.keep_prob_gnn = tf.placeholder(tf.float32)  # dropout keep prob
        self.is_training = tf.placeholder(tf.bool)
        self.features_answer_index = tf.placeholder(tf.int32, [None, self.max_step + 1, self.field_size])
        self.target_answers = tf.placeholder(tf.float32, [None, self.max_step])
        self.sequence_lens = tf.placeholder(tf.int32, [None])
        self.hist_neighbor_index = tf.placeholder(tf.int32, [None, self.max_step, self.hist_neighbor_num])
        self.batch_size = tf.shape(self.features_answer_index)[0]
        self.feature_embedding = tf.get_variable("feature_embedding", [self.feature_answer_size, self.embedding_size],initializer=tf.contrib.layers.xavier_initializer())
        self.stdv = 1.0 / math.sqrt(self.hidden_neurons[-1])    
        self.W_in = tf.get_variable('W_in', shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.embedding_size, self.embedding_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.embedding_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w1 = tf.get_variable('nasr_w1', [self.embedding_size, self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.embedding_size, self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasr_v', [self.embedding_size, self.embedding_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.embedding_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.nasr_w3 =tf.get_variable('nasr_w3', [self.embedding_size*2, self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)
        self.build_model()

    def build_model(self):
        hidden_size = self.hidden_neurons[-1]
        select_feature_index = tf.gather(self.features_answer_index, self.select_index, axis=-1)
        select_size = len(self.select_index)
        questions_index = select_feature_index[:, :-1, 1]
        next_questions_index =select_feature_index[:,1:,1]
        skill_index = select_feature_index[:,:-1,0]
        next_skill_index =select_feature_index[:,1:,0]
        input_answers_index = select_feature_index[:,:-1,-1]
        self.input_questions_embedding = tf.nn.embedding_lookup(self.feature_embedding, questions_index) #[batch_size,seq_len,d]
        self.next_questions_embedding = tf.nn.embedding_lookup(self.feature_embedding,next_questions_index) #[batch_size,seq_len,select_size-1,d]
        self.input_skills_embedding = tf.nn.embedding_lookup(self.feature_embedding, skill_index) #[batch_size,seq_len,d]
        self.next_skills_embedding = tf.nn.embedding_lookup(self.feature_embedding,next_skill_index) #[batch_size,seq_len,select_size-1,d]
        input_answers_embedding = tf.nn.embedding_lookup(self.feature_embedding,input_answers_index) #[batch_size,seq_len,1,d]
        
        #HRG Embedding Module
        #gcn
        if self.n_hop>0:
            input_neighbors = self.get_neighbors(self.n_hop,questions_index)##[[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]
            aggregate_embedding,self.aggregators = self.aggregate(input_neighbors, self.input_questions_embedding)
            next_input_neighbors = self.get_neighbors(self.n_hop,next_questions_index)##[[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]
            next_aggregate_embedding,self.aggregators = self.aggregate(next_input_neighbors, self.next_questions_embedding)
            feature_emb_size =  self.embedding_size
            feature_trans_embedding  = tf.reshape(tf.layers.dense(tf.reshape(aggregate_embedding[0],[-1,feature_emb_size]),hidden_size, activation = tf.nn.relu, name = 'feature_layer', reuse = False), [-1,self.max_step, hidden_size]) #[batch_size,max_step,hidden_size]
            next_trans_embedding  = tf.reshape(tf.layers.dense(tf.reshape(next_aggregate_embedding[0],[-1,feature_emb_size]),hidden_size, activation = tf.nn.relu, name = 'feature_layer', reuse = True), [-1,self.max_step, hidden_size]) #[batch_size,max_step,hidden_size]
        else:
            feature_emb_size =  self.embedding_size
            feature_trans_embedding  = tf.reshape(tf.layers.dense(tf.reshape(self.input_questions_embedding,[-1,feature_emb_size]),hidden_size, activation = tf.nn.relu, name = 'feature_layer', reuse = False), [-1,self.max_step, hidden_size]) #[batch_size,max_step,hidden_size]
            next_trans_embedding  = tf.reshape(tf.layers.dense(tf.reshape(self.next_questions_embedding,[-1,feature_emb_size]),hidden_size, activation = tf.nn.relu, name = 'feature_layer', reuse = True), [-1,self.max_step, hidden_size]) #[batch_size,max_step,hidden_size]
            input_neighbors = self.get_neighbors(1,
                                                 questions_index)  ##[[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]
            next_input_neighbors = self.get_neighbors(1,
                                                      next_questions_index)  ##[[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]
            next_aggregate_embedding = [next_trans_embedding,tf.reshape(tf.gather(self.feature_embedding, tf.reshape(next_input_neighbors[-1], [-1])),
                                            [self.batch_size, self.max_step, -1, self.embedding_size])]
            aggregate_embedding = [feature_trans_embedding,tf.reshape(tf.gather(self.feature_embedding, tf.reshape(input_neighbors[-1], [-1])),
                                            [self.batch_size, self.max_step, -1, self.embedding_size])]
        input_fa_embedding = tf.reshape(tf.concat([feature_trans_embedding,input_answers_embedding],-1),[-1,hidden_size+self.embedding_size]) #embedding_size*2
        input_trans_embedding = tf.reshape(tf.layers.dense(input_fa_embedding, hidden_size),
                                            [-1, self.max_step, hidden_size])
        
        if self.args.model == "ssei":
            if self.args.sim_emb == "skill_emb":
                self.hist_neighbors_features = self.hist_neighbor_sampler1(self.input_skills_embedding,
                                                                           self.next_skills_embedding,
                                                                           input_trans_embedding)  # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
            elif self.args.sim_emb == "question_emb":
                self.hist_neighbors_features = self.hist_neighbor_sampler1(self.input_questions_embedding,
                                                                           self.next_questions_embedding,
                                                                           input_trans_embedding)  # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
            else:
                self.hist_neighbors_features = self.hist_neighbor_sampler1(feature_trans_embedding,
                                                                           next_trans_embedding,
                                                                           input_trans_embedding)  # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
        else:
            if self.args.sim_emb == "skill_emb":
                self.hist_neighbors_features = self.hist_neighbor_sampler1(self.input_skills_embedding,
                                                                           self.next_skills_embedding,
                                                                           output_series)  # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
            elif self.args.sim_emb == "question_emb":
                self.hist_neighbors_features = self.hist_neighbor_sampler1(self.input_questions_embedding,
                                                                           self.next_questions_embedding,
                                                                           output_series)  # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
            else:
                self.hist_neighbors_features = self.hist_neighbor_sampler1(feature_trans_embedding,
                                                                           next_trans_embedding,
                                                                           output_series)  # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]

        #SG Embedding Module
        #GGNN
        output_series = []
        fin_state = tf.nn.embedding_lookup(self.feature_embedding,input_answers_index[:,0])
        cell = tf.nn.rnn_cell.GRUCell(self.embedding_size, dtype=tf.float32)
        with tf.variable_scope('gru'):
            for i in range(self.max_step):
                question_state = tf.nn.embedding_lookup(self.feature_embedding, questions_index[:,i])
                answer_state = tf.nn.embedding_lookup(self.feature_embedding, input_answers_index[:,i])
                in_state = question_state + answer_state + input_trans_embedding[:, i, :]
                fin_state_in = tf.reshape(tf.matmul(in_state, self.W_in) + self.b_in, [ -1, self.embedding_size])
                fin_state_out = tf.reshape(tf.matmul(in_state, self.W_out) + self.b_out, [-1, self.embedding_size])
                av = tf.concat([fin_state_in , fin_state_out], axis=-1)
                av = av + tf.concat([in_state, in_state], axis=-1)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.embedding_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.embedding_size]))
                output_series.append(state_output)
            output_series = tf.reshape(tf.concat(output_series, 1), [-1, self.max_step, self.embedding_size])

        #Self-attention with FM Module
        E_ansewring_states = self.hist_neighbor_sampler(input_trans_embedding)

        #Prediction Module
        if self.next_neighbor_num!=0:
            Nn = self.next_neighbor_sampler(next_aggregate_embedding)  # [batch_size,max_step,N+1,embedding_size]
            Nn = tf.concat([tf.expand_dims(next_trans_embedding,2),Nn],-2)
            next_neighbor_num = self.next_neighbor_num+1
        else:
            Nn = tf.expand_dims(next_trans_embedding, 2)
            next_neighbor_num = 1
        if self.hist_neighbor_num != 0:
            Nh = tf.concat([tf.expand_dims(output_series, 2), self.hist_neighbors_features+E_ansewring_states],
                           2)  # [self.batch_size,max_step,M+1,feature_trans_size]]
            logits = tf.reduce_sum(tf.expand_dims(Nh, 3) * tf.expand_dims(Nn, 2),
                                   axis=4)  # [-1,max_step,Nh,1,emb_size]*[-1,max_step,1,Nn,emb_size]
            logits = tf.reshape(logits, [-1, self.max_step, (
                        self.hist_neighbor_num + 1) * next_neighbor_num])  # ====>[batch_size,max_step,Nu*Nv]
        else:
            Nh = tf.expand_dims(output_series, 2)  # [self.batch_size,max_step,1,feature_trans_size]
            logits = tf.reduce_sum(tf.expand_dims(Nh, 3) * tf.expand_dims(Nn, 2),
                                   axis=4)  # [-1,max_step,Nh,1,emb_size]*[-1,max_step,1,Nn,emb_size]
            logits = tf.reshape(logits,
                                [-1, self.max_step, 1 * next_neighbor_num])  # ====>[batch_size,max_step,Nu*Nv]
        with tf.variable_scope('ni'):
            w1 = tf.get_variable('atn_weights_1',[hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable('atn_weights_2',[hidden_size, 1],initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable('atn_bias_1',[1],initializer=tf.zeros_initializer())
            b2 = tf.get_variable('atn_bias_2',[1],initializer=tf.zeros_initializer())
        if select_size > 3:
            f1 = tf.reshape(tf.matmul(tf.reshape(Nh, [-1, hidden_size]), w1) + b1,
                            [-1, self.max_step, self.hist_neighbor_num + 1, 1])
            f2 = tf.reshape(tf.matmul(tf.reshape(Nn, [-1, hidden_size]), w2) + b2,
                            [-1, self.max_step, 1, next_neighbor_num])
            coefs = tf.nn.softmax(tf.nn.tanh(
                tf.reshape(f1 + f2, [-1, self.max_step, (self.hist_neighbor_num + 1) * next_neighbor_num])))  # temp=10
        else:
            f1 = tf.reshape(tf.matmul(tf.reshape(Nh, [-1, hidden_size]), w1) + b1,
                            [-1, self.max_step, self.hist_neighbor_num + 1, 1])
            f2 = tf.reshape(tf.matmul(tf.reshape(Nn, [-1, hidden_size]), w2) + b2,
                            [-1, self.max_step, 1, next_neighbor_num])
            coefs = tf.nn.softmax(tf.nn.tanh(
                tf.reshape(f1 + f2, [-1, self.max_step, (self.hist_neighbor_num + 1) * next_neighbor_num])))  # temp=10
        self.logits = tf.reduce_sum(logits * coefs, axis=-1)
        self.flat_target_logits = flat_target_logits = tf.reshape(self.logits, [-1])
        self.flat_target_correctness = tf.reshape(self.target_answers, [-1])
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits, [-1, self.max_step]))
        self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.int32)
        self.filling_seqs = tf.cast(tf.sequence_mask(self.sequence_lens - 1, self.max_step),
                                    dtype=tf.float32)  # [batch_size,seq_len]
        index = tf.where(tf.not_equal(tf.reshape(self.filling_seqs, [-1]), tf.constant(0, dtype=tf.float32)))
        clear_flat_target_logits = tf.gather(self.flat_target_logits, index)
        clear_flat_target_correctness = tf.gather(self.flat_target_correctness, index)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=clear_flat_target_correctness,
                                                                          logits=clear_flat_target_logits))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        trainable_vars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 50)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                               beta1=0.9, beta2=0.999, epsilon=1e-8). \
            minimize(self.loss, global_step=self.global_step)
        self.finalTrainOp = tf.train.AdamOptimizer(learning_rate=0.8,
                                               beta1=0.9, beta2=0.999, epsilon=1e-8). \
            minimize(self.loss, global_step=self.global_step)
        print("initialize complete")

    #Self-attention with FM Module
    def hist_neighbor_sampler(self,input_embedding):
        temp = []
        x = tf.get_variable('xita', shape=[self.max_step], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        xt1 = tf.get_variable('xt1', shape=[self.max_step, self.max_step], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        xt2 = tf.get_variable('xt2', shape=[self.max_step, self.max_step], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        for i in range(self.hidden_neurons[-1]):
            temp.append(tf.matmul(tf.exp(tf.matmul((input_embedding[:,:,i]-input_embedding[:,:,0]),xt1)),xt2)+x)
        input_embedding = tf.reshape(tf.stack(temp),[self.batch_size,self.max_step,self.hidden_neurons[-1]])
        zero_embeddings = tf.expand_dims(tf.zeros([self.batch_size,self.hidden_neurons[-1]],dtype=tf.float32),1)#[batch_size,1,hidden_size]
        input_embedding = tf.concat([input_embedding,zero_embeddings],1)#[batch_size,max_step+1,hidden_size]
        #input_embedding:[batch_size,max_step,fa_trans_size]
        b = tf.get_variable('bias', shape=[self.max_step+1, self.hidden_neurons[-1]], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        K = tf.get_variable('K', shape=[self.hidden_neurons[-1], self.hidden_neurons[-1]], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        Q = tf.get_variable('Q', shape=[self.hidden_neurons[-1], self.hidden_neurons[-1]], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        V = tf.get_variable('V', shape=[self.hidden_neurons[-1], self.hidden_neurons[-1]], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        EK= tf.matmul(input_embedding,K)+b
        EQ = tf.matmul(input_embedding,Q)+b
        EV = tf.matmul(input_embedding,V)+b
        A = tf.matmul(EQ,EK,transpose_b=True) / tf.sqrt(float(self.hidden_neurons[-1]))
        O = tf.matmul(A,EV)
        temp_hist_index = tf.reshape(self.hist_neighbor_index,[-1,self.max_step*self.hist_neighbor_num]) #[self.batch_size, max_step*M]
        hist_neighbors_features =tf.reshape(tf.batch_gather(O,temp_hist_index),[-1,self.max_step,self.hist_neighbor_num,input_embedding.shape[-1]])
        return hist_neighbors_features

    def hist_neighbor_sampler1(self, input_q_emb, next_q_emb, qa_emb):#sample based on question similarity
        #next_q_emb:[batch_size,ms,emb_size]
        mold_nextq = tf.sqrt(tf.reduce_sum(next_q_emb*next_q_emb,-1))#[bs,ms]
        next_q_emb = tf.expand_dims(next_q_emb,2)
        mold_inputq = tf.sqrt(tf.reduce_sum(input_q_emb*input_q_emb,-1))#[bs,ms]
        input_q_emb = tf.expand_dims(input_q_emb,1)
        q_similarity = tf.reduce_sum(next_q_emb*input_q_emb,-1)#[batch_size,ms,ms]
        molds = tf.expand_dims(mold_nextq,2)*tf.expand_dims(mold_inputq,1)#[bs,ms,ms]
        q_similarity = q_similarity/molds
        zero_embeddings = tf.expand_dims(tf.zeros([self.batch_size,self.hidden_neurons[-1]],dtype=tf.float32),1)#[batch_size,1,hidden_size]
        qa_emb = tf.concat([qa_emb, zero_embeddings], 1)#[batch_size,max_step+1,hidden_size]
        #mask future position
        seq_mask = tf.range(1,self.max_step+1)
        similarity_seqs = tf.tile(tf.expand_dims(tf.cast(tf.sequence_mask(seq_mask, self.max_step),
                                    dtype=tf.float32),0),[self.batch_size,1,1])  # [batch_size,ms,ms]
        q_similarity = q_similarity*similarity_seqs #only history q non zero# [batch_size,ms,ms]
        #setting lower similarity bount
        condition = tf.greater(q_similarity,self.args.att_bound)
        #condition = tf.greater(q_similarity,0.7)
        q_similarity = tf.where(condition,q_similarity,tf.zeros([self.batch_size,self.max_step,self.max_step]))
        self.q_similarity = q_similarity
        temp_hist_index = tf.nn.top_k(q_similarity, self.hist_neighbor_num)[1]# [batch_size,ms,hist_num]
        self.hist_attention_value = tf.nn.top_k(q_similarity, self.hist_neighbor_num)[0]# [batch_size,ms,hist_num]
        #q_similarity[temp_hist_index]>0
        temp_hist_index = tf.where(self.hist_attention_value>0,temp_hist_index,self.hist_neighbor_index)
        temp_hist_index = tf.reshape(temp_hist_index,[-1,self.max_step*self.hist_neighbor_num])
        hist_neighbors_features =tf.reshape(tf.batch_gather(qa_emb, temp_hist_index), [-1, self.max_step, self.hist_neighbor_num, qa_emb.shape[-1]])
        return hist_neighbors_features


    def next_neighbor_sampler(self,aggregate_embedding):
        temp_emb = tf.reshape(aggregate_embedding[1],[-1,self.question_neighbor_num,self.embedding_size])
        temp_emb = tf.transpose(temp_emb, [1, 0, 2])
        temp_emb = tf.transpose(
            tf.gather(temp_emb, tf.random.shuffle(tf.range(tf.shape(temp_emb)[0]))), [1, 0, 2])
        if self.question_neighbor_num>=self.next_neighbor_num:
            next_neighbors_embedding = tf.reshape(temp_emb[:,:self.next_neighbor_num,:],[self.batch_size,self.max_step,self.next_neighbor_num,self.embedding_size])
        else:
            tile_neighbor_embedding = tf.tile(temp_emb,[1, -(-self.next_neighbor_num // tf.shape(temp_emb)[0]), 1])
            next_neighbors_embedding = tf.reshape(tile_neighbor_embedding[:,:self.next_neighbor_num,:],[self.batch_size,self.max_step,self.next_neighbor_num,self.embedding_size])
        return next_neighbors_embedding

    def get_neighbors(self,n_hop, question_index):
        # question_index:[batch_size,seq_len]
        seeds = [question_index]  # [[batch_size,seq_len],[batch_size,seq_len,question_neighbor_num],batch_size,seq_len,question_neighbor_num,
        for i in range(n_hop):
            if i % 2 == 0:
                neighbor = tf.reshape(tf.gather(self.question_neighbors, tf.reshape(seeds[i], [-1])),
                                      [-1, self.max_step, self.question_neighbor_num])

            else:
                neighbor = tf.reshape(tf.gather(self.question_neighbors, tf.reshape(seeds[i], [-1])),
                                      [-1, self.max_step, self.skill_neighbor_num])
            seeds.append(neighbor)  # [batch_size,seq_len,neighbor_num],[batch_size,seq_len,neighbor_num*neighbor_num]
        return seeds

    def aggregate(self, input_neighbors, input_questions_embedding):
        # [[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]]
        sq_neighbor_vectors = []
        for hop_i, neighbors in enumerate(input_neighbors):
            if hop_i % 2 == 0:  # question
                temp_neighbors = tf.reshape(tf.gather(self.feature_embedding, tf.reshape(neighbors, [-1])),
                                            [self.batch_size, self.max_step, -1, self.embedding_size])
                sq_neighbor_vectors.append(temp_neighbors)
            else:  # skill
                temp_neighbors = tf.reshape(tf.gather(self.feature_embedding, tf.reshape(neighbors, [-1])),
                                            [self.batch_size, self.max_step, -1, self.embedding_size])
                sq_neighbor_vectors.append(temp_neighbors)
        aggregators = []
        for i in range(self.n_hop):
            if i == self.n_hop - 1:
                aggregator = self.aggregator_class(self.batch_size, self.max_step, self.embedding_size, act=tf.nn.tanh,
                                                   dropout=self.keep_prob_gnn)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.max_step, self.embedding_size, act=tf.nn.tanh,
                                                   dropout=self.keep_prob_gnn)
            aggregators.append(aggregator)
            for hop in range(self.n_hop - i):  # aggregate from outside to inside#layer
                if hop % 2 == 0:
                    shape = [self.batch_size, self.max_step, -1, self.question_neighbor_num, self.embedding_size]
                    vector = aggregator(self_vectors=sq_neighbor_vectors[hop],
                                        neighbor_vectors=tf.reshape(sq_neighbor_vectors[hop + 1], shape),
                                        question_embeddings=sq_neighbor_vectors[hop],
                                        )  # [batch_size,seq_len, -1, dim]
                else:
                    shape = [self.batch_size, self.max_step, -1, self.skill_neighbor_num, self.embedding_size]
                    vector = aggregator(self_vectors=sq_neighbor_vectors[hop],
                                        neighbor_vectors=tf.reshape(sq_neighbor_vectors[hop + 1], shape),
                                        question_embeddings=sq_neighbor_vectors[hop],
                                        )  # [batch_size,seq_len, -1, dim]
                sq_neighbor_vectors[hop] = vector
        res = sq_neighbor_vectors  # [[batch_size,max_step,-1,embedding_size]...]
        return res, aggregators

    # step on batch
    def train(self, sess, features_answer_index, target_answers, seq_lens, hist_neighbor_index):
        input_feed = {self.features_answer_index: features_answer_index,
                      self.target_answers: target_answers,
                      self.sequence_lens: seq_lens,
                      self.hist_neighbor_index: hist_neighbor_index,
                      self.is_training: True}
        input_feed[self.keep_prob] = self.dropout_keep_probs[0]
        input_feed[self.keep_prob_gnn] = self.dropout_keep_probs[1]
        bin_pred, pred, train_loss, _, aaaa = sess.run(
            [self.binary_pred, self.pred, self.loss, self.train_op, self.flat_target_correctness], input_feed)
        return bin_pred, pred, train_loss

    def evaluate(self, sess, features_answer_index, target_answers, seq_lens, hist_neighbor_index,evaluate_step):
        input_feed = {self.features_answer_index: features_answer_index,
                      self.target_answers: target_answers,
                      self.sequence_lens: seq_lens,
                      self.hist_neighbor_index: hist_neighbor_index,
                      self.is_training: False}
        input_feed[self.keep_prob] = self.dropout_keep_probs[-1]
        input_feed[self.keep_prob_gnn] = self.dropout_keep_probs[-1]
        bin_pred, pred = sess.run([self.binary_pred, self.pred], input_feed)
        return bin_pred, pred
